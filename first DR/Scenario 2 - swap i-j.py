# -*- coding: utf-8 -*-
"""
Operating Room Scheduling — Step 1 + Step 2 (Iterative Dispatching Rule)
Scenario 2: Surgeon can change room within the same shift
Author: Joana
"""

from pathlib import Path
import re
import ast
import pandas as pd

# ------------------------------
# PARAMETERS
# ------------------------------
DATA_FILE = "Instance_C1_30.dat"

C_PER_SHIFT = 360   # minutes per shift (6h * 60)
CLEANUP = 17        # cleaning time 

ALPHA1 = 0.70 # priority
ALPHA2 = 0.1  # waited days
ALPHA3 = 0.1 # deadline closeness
ALPHA4 = 0.1 # feasible blocks

ALPHA5 = 1/3
ALPHA6 = 1/3
ALPHA7 = 1/3

TOLERANCE = 15  #we allow 15 minutes delays after end time of each block
ROOM_CHANGE_TIME = 5  #we assume that a surgeon changing rooms takes 5 minutes

# ------------------------------
# FILE READING FUNCTIONS
# ------------------------------
text = Path(DATA_FILE).read_text(encoding="utf-8")

def get_int(name, alt_name=None):
    m = re.search(rf'int\s+{name}\s*=\s*(\d+)', text)
    if not m and alt_name:
        m = re.search(rf'int\s+{alt_name}\s*=\s*(\d+)', text)
    if not m:
        raise ValueError(f"Integer '{name}' not found in {DATA_FILE}")
    return int(m.group(1))

def get_array(name):
    m = re.search(rf'{name}\s*=\s*(\[[\s\S]*?\])\s*;', text, re.DOTALL)
    if not m:
        raise ValueError(f"Array '{name}' not found in {DATA_FILE}")
    return ast.literal_eval(m.group(1))


# ------------------------------
# READ DATA FROM FILE
# ------------------------------
n_patients = get_int("NumberPatients")
durations  = get_array("Duration")
priorities = get_array("Priority")
waitings   = get_array("Waiting")
surgeons   = get_array("Surgeon")

n_rooms = get_int("NumberOfRooms")
n_days  = get_int("NumberOfDays")
block_av = get_array("BlockAvailability")

n_surgeons = get_int("NumberSurgeons", alt_name="NumberOfSurgeons")
surg_av = get_array("SurgeonAvailability")


# ------------------------------
# DATAFRAMES
# ------------------------------
# Patients

df_patients = pd.DataFrame({
    "patient_id": range(1, n_patients + 1),
    "duration": durations,
    "priority": priorities,
    "waiting": waitings,
    "surgeon_id": surgeons
})

# Rooms
rows = []
for r in range(n_rooms):
    for d in range(n_days):
        for shift in (1, 2):  # 1=AM, 2=PM
            available = int(block_av[d][r][shift - 1])
            rows.append({"room": r + 1, "day": d + 1, "shift": shift, "available": available})
df_rooms = pd.DataFrame(rows)

# Surgeons
rows = []
for s in range(n_surgeons):
    for d in range(n_days):
        for shift in (1, 2):
            availability = int(surg_av[s][d][shift - 1])
            rows.append({"surgeon_id": s + 1, "day": d + 1, "shift": shift, "available": availability})
df_surgeons = pd.DataFrame(rows)

# ------------------------------
# STEP 2 SUPPORT FUNCTIONS
# ------------------------------

def feasible_blocks_step2(patient_row):
    sid = int(patient_row["surgeon_id"])
    need = int(patient_row["duration"]) + CLEANUP

    # últimos blocos do cirurgião no mesmo dia/shift
    prev_assigns = df_assignments[
        (df_assignments["surgeon_id"] == sid) &
        (df_assignments["day"] == patient_row.get("day", 0)) &
        (df_assignments["shift"] == patient_row.get("shift", 0))
    ]
    last_room = prev_assigns["room"].iloc[-1] if not prev_assigns.empty else None

    # capacidade por bloco (inclui free_min)
    cap_ok = df_capacity[df_capacity["available"]==1].copy()

    # ADICIONAR cap_min (porque df_capacity não tem esta coluna)
    cap_ok["cap_min"] = C_PER_SHIFT

    # need = duração + cleanup + mudança de sala se aplicável
    cap_ok["need"] = need
    #cap_ok.loc[cap_ok["room"] != last_room, "need"] += ROOM_CHANGE_TIME

    # used_min do bloco ATUAL
    cap_ok["used_min"] = cap_ok["cap_min"] - cap_ok["free_min"]

    # juntar disponibilidade do cirurgião
    surg_ok = df_surgeons[
        (df_surgeons["surgeon_id"] == sid) & (df_surgeons["available"] == 1)
    ][["day","shift"]]

    cand = cap_ok.merge(surg_ok, on=["day","shift"], how="inner")

    # juntar carga do cirurgião nesse dia/shift
    sload = df_surgeon_load[df_surgeon_load["surgeon_id"] == sid][["day","shift","used_min"]]
    sload = sload.rename(columns={"used_min": "surg_used"})
    cand = cand.merge(sload, on=["day","shift"], how="left").fillna({"surg_used": 0})

    # RESTRIÇÃO PRINCIPAL: respeitar capacidade + tolerância
    cand = cand[(cand["used_min"] + cand["need"]) <= C_PER_SHIFT + TOLERANCE]
    cand = cand[(cand["surg_used"] + cand["need"]) <= C_PER_SHIFT + TOLERANCE]

    # continuidade: 1 se o cirurgião já operou nesse bloco
    if len(df_assignments) > 0:
        cont = df_assignments[df_assignments["surgeon_id"] == sid][["room","day","shift"]].copy()
        cont["continuity"] = 1
        cand = cand.merge(cont, on=["room","day","shift"], how="left")
        cand["continuity"] = cand["continuity"].fillna(0).astype(int)
    else:
        cand["continuity"] = 0

    return cand

#CALCULATING PATIENT SCORE

def score_block_for_patient(cand_df, patient_row, n_days):   #step 2 of dispatching rule
    """Compute W_block for each candidate."""
    need = int(patient_row["duration"]) + CLEANUP
    day_max = max(1, n_days - 1)
    df = cand_df.copy()
    df["free_after"] = (df["free_min"] - need).clip(lower=0)
    df["term_fit"]   = 1.0 - (df["free_after"] / C_PER_SHIFT)
    df["term_early"] = 1.0 - ((df["day"] - 1) / day_max)
    df["term_cont"]  = df["continuity"].astype(float)
    df["W_block"] = ALPHA5 * df["term_fit"] + ALPHA6 * df["term_early"] + ALPHA7 * df["term_cont"]
    return df.sort_values("W_block", ascending=False)


def commit_assignment(patient_row, best_row, iteration, w_patient=None, w_block=None):
    pid = int(patient_row["patient_id"])
    sid = int(patient_row["surgeon_id"])
    dur_need = int(patient_row["duration"]) + CLEANUP
    r, d, sh = int(best_row["room"]), int(best_row["day"]), int(best_row["shift"])

    # calcular tempo extra se cirurgião muda de sala
    previous_assigns = df_assignments[
        (df_assignments["surgeon_id"] == sid) & 
        (df_assignments["day"] == d) &
        (df_assignments["shift"] == sh)
    ]
    if not previous_assigns.empty:
        last_room = previous_assigns.iloc[-1]["room"]
        changed_room = (last_room != r)
    else:
            changed_room = False

    # update capacity
    idx = (
        (df_capacity["room"] == r) &
        (df_capacity["day"] == d) &
        (df_capacity["shift"] == sh)
    )
    df_capacity.loc[idx, "free_min"] -= dur_need

    # update surgeon load
    idx_s = (
        (df_surgeon_load["surgeon_id"] == sid) &
        (df_surgeon_load["day"] == d) &
        (df_surgeon_load["shift"] == sh)
    )
    df_surgeon_load.loc[idx_s, "used_min"] += dur_need

    # record assignment
    df_assignments.loc[len(df_assignments)] = {
        "patient_id": pid,
        "room": r,
        "day": d,
        "shift": sh,
        "used_min": dur_need,
        "surgeon_id": sid,
        "iteration": int(iteration),
        "W_patient": float(w_patient) if w_patient is not None else None,
        "W_block": float(w_block) if w_block is not None else None,
    }


def deadline_limit_from_priority(p):
    return 3 if p == 3 else (15 if p == 2 else (90 if p == 1 else 270))

def deadline_term(priority, waited):
    lim = deadline_limit_from_priority(priority)
    if lim is None:
        return 0.0
    days_left = max(0, lim - waited)
    return 1.0 - (days_left / lim)


# --------------------------------------------
# FEASIBILITY & EVALUATION FUNCTIONS
# --------------------------------------------

def feasibility_metrics(assignments, df_rooms, df_surgeons, patients, C_PER_SHIFT):
    # --- bases de capacidade/availability por bloco e por cirurgião
    rooms_base = df_rooms[["room", "day", "shift", "available"]].copy()
    rooms_base["cap_min"] = rooms_base["available"] * C_PER_SHIFT

    surg_base = df_surgeons[["surgeon_id", "day", "shift", "available"]].drop_duplicates()

    # --- uso por bloco a partir das assignações
    if len(assignments):
        used_by_block = (assignments.groupby(["room", "day", "shift"], as_index=False)
                         .agg(used_min=("used_min", "sum")))
    else:
        used_by_block = rooms_base[["room","day","shift"]].copy()
        used_by_block["used_min"] = 0

    rooms_join = rooms_base.merge(used_by_block, on=["room","day","shift"], how="left").fillna({"used_min":0})

    # 1) excesso de tempo por bloco (não pode haver)
    rooms_join["excess_min"] = (rooms_join["used_min"] - rooms_join["cap_min"]).clip(lower=0)
    excess_block_min = int(rooms_join["excess_min"].sum())

    # 2) atribuições em blocos fechados (availability=0)
    bad_block_assigns = assignments.merge(rooms_base, on=["room","day","shift"], how="left")
    block_unavailable_viol = int((bad_block_assigns["available"].fillna(0) == 0).sum())

    # 3) disponibilidade do cirurgião (não pode operar se available=0)
    ass_with_surg = assignments.merge(
        surg_base, on=["surgeon_id","day","shift"], how="left", suffixes=("","_s")
    )
    surg_unavailable_viol = int((ass_with_surg["available"].fillna(0) == 0).sum())

    # 4) estouro de tempo por cirurgião em cada (day,shift) (limite = C_PER_SHIFT)
    if len(assignments):
        sload = (assignments.groupby(["surgeon_id","day","shift"], as_index=False)
                 .agg(used_min=("used_min","sum")))
        sload["excess_min"] = (sload["used_min"] - C_PER_SHIFT).clip(lower=0)
        excess_surgeon_min = int(sload["excess_min"].sum())
    else:
        excess_surgeon_min = 0

    # 5) pacientes não agendados
    n_unassigned = int(len(patients) - len(assignments))

    # score total de infeasibilidade (minimizar; 0 => solução viável)
    feasibility_score = (
        block_unavailable_viol
        + surg_unavailable_viol
        + excess_block_min
        + excess_surgeon_min
    )

    return {
        "n_unassigned": n_unassigned,
        "block_unavailable_viol": block_unavailable_viol,
        "surg_unavailable_viol": surg_unavailable_viol,
        "excess_block_min": excess_block_min,
        "excess_surgeon_min": excess_surgeon_min,
        "feasibility_score": feasibility_score,
        "rooms_cap_join": rooms_join
    }

def evaluate_schedule(assignments, patients, rooms_free, excess_block_min, weights=(0.6, 0.1, 0.25, 0.05)):
    w1, w2, w3, w4 = weights
    total_patients = len(patients)
    ratio_scheduled = (len(assignments) / total_patients) if total_patients else 0.0

    util_rooms = float(rooms_free.loc[rooms_free["cap_min"] > 0, "utilization"].mean()) if len(rooms_free) else 0.0

    if len(assignments):
        # assignments JÁ tem priority e waiting
        merged = assignments
    
        prio_rate = float((merged["priority"] > 0).mean())
        wmax = float(patients["waiting"].max())
        norm_wait_term = 1.0 - float(merged["waiting"].mean() / wmax) if wmax > 0 else 1.0
    else:
        prio_rate = 0.0
        norm_wait_term = 0.0



    score = (w1*ratio_scheduled + w2*util_rooms + w3*prio_rate + w4*norm_wait_term)-0.001* excess_block_min
    return {"score":float(score), "ratio_scheduled":float(ratio_scheduled),
            "util_rooms":float(util_rooms), "prio_rate":float(prio_rate),
            "norm_wait_term":float(norm_wait_term)}


def generate_neighbor_swap(current_assignments,         #isto aqui já é depois do schedule???
                           df_patients,
                           df_rooms,
                           df_surgeons,
                           C_PER_SHIFT,
                           max_swap_out=10,
                           max_swap_in=3):
    """
    Gera um vizinho da solução atual e devolve também:
      - ids_out: lista de pacientes retirados
      - ids_in_effective: lista de pacientes que foram mesmo adicionados
    """

    # cópia para não estragar o original
    assignments = current_assignments.copy()

    # pacientes atualmente agendados
    scheduled_ids = assignments["patient_id"].unique().tolist()
    if len(scheduled_ids) == 0:
        return assignments, [], []  # não há nada para mexer

    # pacientes não agendados = todos - scheduled
    all_ids = df_patients["patient_id"].unique().tolist()
    unassigned_ids = [pid for pid in all_ids if pid not in scheduled_ids]

    # 1) REMOVER I (até max_swap_out)
    k_out = min(max_swap_out, len(scheduled_ids))
    ids_out = random.sample(scheduled_ids, k_out)
    assignments = assignments[~assignments["patient_id"].isin(ids_out)].copy()

    # 2) ADICIONAR J (até max_swap_in)
    random.shuffle(unassigned_ids)
    ids_in_candidates = unassigned_ids[:max_swap_in]

    ids_in_effective = []   # <- aqui vamos guardar só os que entram MESMO

    for pid in ids_in_candidates:
        prow = df_patients[df_patients["patient_id"] == pid]
        if prow.empty:
            continue
        prow = prow.iloc[0]

        # blocos viáveis dado o assignments ATUAL
        cand_blocks = candidate_blocks_for_patient_in_solution(            #devolve os blocos (room, day, shift) onde este paciente pode ser colocado dada a situação atual
            assignments, prow, df_rooms, df_surgeons, C_PER_SHIFT
        )
        if len(cand_blocks) == 0:
            continue

        chosen = cand_blocks.sample(1).iloc[0]
        room = int(chosen["room"])
        day = int(chosen["day"])
        shift = int(chosen["shift"])
        dur = int(prow["duration"])
        sid = int(prow["surgeon_id"])
        need = dur + CLEANUP

        new_row = {
            "patient_id": pid,
            "room": room,
            "day": day,
            "shift": shift,
            "used_min": need,
            "surgeon_id": sid,
            "iteration": -1,      # marca que veio do ILS
            "W_patient": 0.0,     # podes recalcular se quiseres
            "W_block": 0.0,
        }

        assignments = pd.concat(
            [assignments, pd.DataFrame([new_row])],
            ignore_index=True
        )

        ids_in_effective.append(pid)   # <- este entrou mesmo

    return assignments, ids_out, ids_in_effective



# iterative LOCAL SEARCH NEW
import random

def candidate_blocks_for_patient_in_solution(assignments, patient_row,
                                             df_rooms, df_surgeons, C_PER_SHIFT):
    sid = int(patient_row["surgeon_id"])
    need = int(patient_row["duration"]) + CLEANUP

    # disponibilidade do cirurgião
    surg_ok = df_surgeons[
        (df_surgeons["surgeon_id"] == sid) & (df_surgeons["available"] == 1)
    ][["day","shift"]]

    # capacidade por bloco (a partir do assignments atual)
    rooms_base = df_rooms[["room","day","shift","available"]].copy()
    rooms_base["cap_min"] = rooms_base["available"] * C_PER_SHIFT

    if len(assignments):
        used_by_block = (assignments.groupby(["room","day","shift"], as_index=False)
                         .agg(used_min=("used_min","sum")))
    else:
        used_by_block = rooms_base[["room","day","shift"]].copy()
        used_by_block["used_min"] = 0

    rooms_join = rooms_base.merge(used_by_block, on=["room","day","shift"], how="left").fillna({"used_min": 0})
    rooms_join["free_min"] = (rooms_join["cap_min"] - rooms_join["used_min"]).clip(lower=0)

    cap_ok = rooms_join[(rooms_join["available"] == 1) & (rooms_join["free_min"] >= need)][["room","day","shift","free_min"]]

    cand = surg_ok.merge(cap_ok, on=["day","shift"], how="inner")

    # carga do cirurgião por (day,shift)
    if len(assignments):
        sload = (assignments[assignments["surgeon_id"] == sid]
                 .groupby(["day","shift"], as_index=False)
                 .agg(used_min=("used_min","sum")))
    else:
        sload = pd.DataFrame(columns=["day","shift","used_min"])

    cand = cand.merge(sload, on=["day","shift"], how="left").fillna({"used_min": 0})
    cand = cand[(cand["used_min"] + need) <= C_PER_SHIFT]

    # NOTE: **NO ROOM-LOCK**: we intentionally do NOT restrict to same room
    return cand

def sequence_global_by_surgeon(assignments_enriched, C_PER_SHIFT, CLEANUP, TOLERANCE, ROOM_CHANGE_TIME):
    if assignments_enriched.empty:
        return assignments_enriched.copy()

    df = assignments_enriched.copy()

    df["room"] = df["room"].astype(int)
    df["day"] = df["day"].astype(int)
    df["shift"] = df["shift"].astype(int)
    df["surgeon_id"] = df["surgeon_id"].astype(int)
    df["duration"] = df["duration"].astype(int)

    df = df.reset_index().rename(columns={"index": "orig_idx"})

    start_min = {}
    end_min = {}
    seq_in_block = {}
    scheduled_flag = {}

    for (day, shift), df_ds in df.groupby(["day", "shift"], sort=False):

        # construir filas por sala (como já tinhas)
        room_queues = {}
        for room, df_room in df_ds.groupby("room", sort=False):
            df_room = df_room.copy()
            surg_stats = (
                df_room.groupby("surgeon_id", as_index=False)
                       .agg(total_dur=("duration","sum"),
                            n_cases=("patient_id","count"))
            ).sort_values(["total_dur","n_cases"], ascending=[False,False])

            ordered_list = []
            for _, srow in surg_stats.iterrows():
                sid = srow["surgeon_id"]
                sub = df_room[df_room["surgeon_id"] == sid].copy()
                sub = sub.sort_values("duration", ascending=True)
                ordered_list.append(sub)
            if ordered_list:
                df_room_ordered = pd.concat(ordered_list, ignore_index=True)
                room_queues[room] = list(df_room_ordered["orig_idx"].values)
            else:
                room_queues[room] = []

        # relógios
        t_room = {room: 0 for room in room_queues.keys()}
        t_surg = {sid: 0 for sid in df_ds["surgeon_id"].unique()}
        last_room_for_surg = {sid: None for sid in df_ds["surgeon_id"].unique()}
        seq_counter = {room: 0 for room in room_queues.keys()}

        remaining = sum(len(q) for q in room_queues.values())

        while remaining > 0:
            best = None
            best_start = None

            for room, queue in room_queues.items():
                if not queue:
                    continue

                idx = queue[0]
                row = df.loc[df["orig_idx"] == idx].iloc[0]
                sid = row["surgeon_id"]
                dur = int(row["duration"])

                room_ready = t_room[room]      # sala livre (incluindo limpeza)
                base_surg_ready = t_surg[sid]  # cirurgião livre (incluindo limpeza anterior)

                # --- penalização de mudança de sala ---
                if last_room_for_surg[sid] is not None and last_room_for_surg[sid] != room:
                    surgeon_ready_with_move = base_surg_ready + ROOM_CHANGE_TIME
                else:
                    surgeon_ready_with_move = base_surg_ready

                earliest = max(room_ready, surgeon_ready_with_move)
                end_candidate = earliest + dur + CLEANUP

                if (best is None) or (earliest < best_start - 1e-9):
                    best = (room, idx, sid, dur, end_candidate, earliest)
                    best_start = earliest

            if best is None:
                break

            room, idx, sid, dur, end_candidate, earliest = best

            # limite do turno + tolerância
            if end_candidate > C_PER_SHIFT + TOLERANCE + 1e-6:
                scheduled_flag[idx] = 0
                room_queues[room].pop(0)
                remaining -= 1
                continue

            start = earliest
            end = end_candidate

            start_min[idx] = start
            end_min[idx] = end
            seq_counter[room] += 1
            seq_in_block[idx] = seq_counter[room]
            scheduled_flag[idx] = 1

            t_room[room] = end
            t_surg[sid] = end
            last_room_for_surg[sid] = room

            room_queues[room].pop(0)
            remaining -= 1

    df["start_min"] = df["orig_idx"].map(start_min)
    df["end_min"] = df["orig_idx"].map(end_min)
    df["seq_in_block"] = df["orig_idx"].map(seq_in_block)
    df["scheduled_by_seq"] = df["orig_idx"].map(scheduled_flag).fillna(0).astype(int)

    return df.drop(columns=["orig_idx"])



def build_room_free_from_assignments(assignments, df_rooms, C_PER_SHIFT):
    # base: capacidade dos blocos
    rooms_base = df_rooms[["room", "day", "shift", "available"]].copy()
    rooms_base["cap_min"] = rooms_base["available"] * C_PER_SHIFT

    # uso por bloco
    if len(assignments):
        used_by_block = (
            assignments.groupby(["room", "day", "shift"], as_index=False)
                       .agg(used_min=("used_min", "sum"))
        )
    else:
        used_by_block = rooms_base[["room", "day", "shift"]].copy()
        used_by_block["used_min"] = 0

    rooms_join = rooms_base.merge(
        used_by_block, on=["room", "day", "shift"], how="left"
    ).fillna({"used_min": 0})

    rooms_join["free_min"] = (rooms_join["cap_min"] - rooms_join["used_min"]).clip(lower=0)
    rooms_join["utilization"] = rooms_join.apply(
        lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0.0,
        axis=1
    )

    rooms_join = rooms_join.sort_values(["room", "day", "shift"]).reset_index(drop=True)
    return rooms_join

def build_surgeon_free_from_assignments(assignments, df_surgeons, C_PER_SHIFT):
    surg_base = df_surgeons[["surgeon_id", "day", "shift", "available"]].drop_duplicates()

    if len(assignments):
        sload = (
            assignments.groupby(["surgeon_id", "day", "shift"], as_index=False)
                       .agg(used_min=("used_min", "sum"))
        )
    else:
        sload = surg_base[["surgeon_id", "day", "shift"]].copy()
        sload["used_min"] = 0

    surg_join = surg_base.merge(
        sload, on=["surgeon_id", "day", "shift"], how="left"
    ).fillna({"used_min": 0})

    surg_join["cap_min"] = surg_join["available"] * C_PER_SHIFT
    surg_join["free_min"] = (surg_join["cap_min"] - surg_join["used_min"]).clip(lower=0)

    surg_join["utilization"] = surg_join.apply(
        lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0.0,
        axis=1
    )

    surg_join = surg_join.sort_values(["surgeon_id", "day", "shift"]).reset_index(drop=True)
    return surg_join

# ------------------------------
# INITIAL PLANNING STATE
# ------------------------------
df_capacity = df_rooms.copy()
df_capacity["free_min"] = df_capacity["available"].apply(
    lambda a: C_PER_SHIFT if a == 1 else 0
)

df_assignments = pd.DataFrame(columns=[
    "patient_id", "room", "day", "shift",
    "used_min", "surgeon_id", "iteration",
    "W_patient", "W_block"
])

df_surgeon_load = df_surgeons[["surgeon_id", "day", "shift"]].drop_duplicates().assign(used_min=0)


# ------------------------------
# ITERATIVE LOOP: Step 1 → Step 2 → commit → repeat
# ------------------------------
remaining = df_patients.copy()
iteration = 0

while True:
    iteration += 1

    # ---- Step 1: dynamic feasible blocks per patient (uses current capacity & surgeon load) ----
    df_surg_open = df_surgeons[
        df_surgeons["available"] == 1
    ][["surgeon_id", "day", "shift"]].drop_duplicates()

    df_cap_open  = df_capacity[
        df_capacity["available"] == 1
    ][["room", "day", "shift", "free_min"]].drop_duplicates()

    df_sload     = df_surgeon_load[
        ["surgeon_id", "day", "shift", "used_min"]
    ].drop_duplicates()
    
    df_pmini = remaining[[
        "patient_id", "surgeon_id", "duration", "priority", "waiting"
    ]].copy()

    if df_pmini.empty:
        print("\nNo remaining patients — stopping.")
        break

    df_pmini["need"] = df_pmini["duration"] + CLEANUP
    
    # (surgeon availability)
    df_p_time = df_pmini.merge(df_surg_open, on="surgeon_id", how="inner")
    
    # (join current room capacity)
    df_p_cap = df_p_time.merge(df_cap_open, on=["day", "shift"], how="inner")
    
    # (current surgeon load per (day,shift))
    df_p_cap = df_p_cap.merge(
        df_sload, on=["surgeon_id", "day", "shift"], how="left"
    ).fillna({"used_min": 0})
    
    # keep only blocks that can host the case now
    df_p_blocks = df_p_cap[
    ((df_p_cap["used_min"] + df_p_cap["need"]) <= C_PER_SHIFT + TOLERANCE) #NEW
    ]
    
    df_p_blocks = df_p_blocks.copy()
    df_p_blocks["overflow_min"] = (
        (df_p_blocks["used_min"] + df_p_blocks["need"]) - C_PER_SHIFT
    ).clip(lower=0)


    # how many time goes over the C_PER_SHIFT (0 if it doesn't) NEW
    df_p_blocks["overflow_min"] = (
        (df_p_cap["used_min"] + df_p_cap["need"]) - C_PER_SHIFT
    ).clip(lower=0)
    
    # count feasible blocks per patient
    df_feas_count = (
        df_p_blocks.groupby("patient_id", as_index=False)
                   .agg(feasible_blocks=("room", "count"))
    )
    
    step1 = df_pmini.merge(
        df_feas_count, on="patient_id", how="left"
    ).fillna({"feasible_blocks": 0})

    # ---- Step 1: stop if nobody has feasible blocks
    if step1["feasible_blocks"].fillna(0).max() == 0:
        #print("\nNo more schedulable patients under Step-1 filters.")
        break

    # ---- Step 1 scoring
    Pmax = max(step1["priority"].max(), 1)
    Wmax = max(step1["waiting"].max(), 1)
    step1["term_priority"] = step1["priority"] / Pmax
    step1["term_waiting"]  = step1["waiting"] / Wmax
    step1["term_deadline"] = step1.apply(
        lambda r: deadline_term(r["priority"], r["waiting"]), axis=1
    )
    step1["term_scarcity"] = 1.0 / (1.0 + step1["feasible_blocks"])
    step1["W_patient"] = (
          ALPHA1 * step1["term_priority"]
        + ALPHA2 * step1["term_waiting"]
        + ALPHA3 * step1["term_deadline"]
        + ALPHA4 * step1["term_scarcity"]
    )

    # ======== varrer o ranking inteiro ========
    made_assignment = False
    step1_sorted = step1.sort_values("W_patient", ascending=False)

    for _, patient_row in step1_sorted.iterrows():
        # ---- Step 2: feasible blocks with current state
        cand_blocks = feasible_blocks_step2(patient_row)
        if cand_blocks.empty:
            # não remover o paciente — apenas tentar o próximo do ranking
            continue

        # score and select best block
        scored = score_block_for_patient(cand_blocks, patient_row, n_days=n_days)
        best_block = scored.iloc[0]

        # commit assignment and store iteration
        patient_row = patient_row.copy()
        patient_row["iteration"] = iteration
        commit_assignment(
            patient_row,
            best_block,
            iteration=iteration,
            w_patient=float(patient_row["W_patient"]),
            w_block=float(best_block["W_block"])
        )

        # remove scheduled patient from remaining
        remaining = remaining[
            remaining["patient_id"] != int(patient_row["patient_id"])
        ]

        # progress log
        #print(
        #    f"Iter {iteration:02d}: "
        #    f"Assign P{int(patient_row['patient_id'])} → "
        #    f"(Room={int(best_block['room'])}, Day={int(best_block['day'])}, Shift={int(best_block['shift'])}), "
        #    f"W_patient={patient_row['W_patient']:.4f}, W_block={best_block['W_block']:.3f}"
        #)

        made_assignment = True
        break  # uma atribuição por iteração

    if not made_assignment:
        print("\nNo assignable patients under current Step-1 ranking (all fail Step-2). Stopping.")
        break


# ---------- 2) Assignments enriched (sequencing) ----------
assignments_enriched = df_assignments.merge(
    df_patients[["patient_id", "duration", "priority", "waiting"]],
    on="patient_id",
    how="left"
).sort_values("iteration")

assignments_enriched = sequence_global_by_surgeon(
    assignments_enriched,
    C_PER_SHIFT=C_PER_SHIFT,
    CLEANUP=CLEANUP,
    TOLERANCE=TOLERANCE,
    ROOM_CHANGE_TIME=ROOM_CHANGE_TIME
)

assignments_seq_view = assignments_enriched[
    assignments_enriched["scheduled_by_seq"] == 1
].sort_values(
    ["day", "shift", "room", "seq_in_block", "patient_id"]
).reset_index(drop=True)

# opcional: cirurgias retiradas pelo sequenciador
assignments_dropped_by_seq = assignments_enriched[
    assignments_enriched["scheduled_by_seq"] == 0
].sort_values(
    ["day", "shift", "room", "patient_id"]
).reset_index(drop=True)
    
# ---------- GUARDAR SOLUÇÃO INICIAL (ANTES DO ILS) ----------
initial_assignments_enriched = assignments_enriched.copy()
initial_assignments_seq = assignments_seq_view.copy()

# capacidades iniciais por bloco e por cirurgião
initial_rooms_free = build_room_free_from_assignments(
    assignments=initial_assignments_seq,
    df_rooms=df_rooms,
    C_PER_SHIFT=C_PER_SHIFT
)

initial_surgeons_free = build_surgeon_free_from_assignments(
    assignments=initial_assignments_seq,
    df_surgeons=df_surgeons,
    C_PER_SHIFT=C_PER_SHIFT
)

# feasibility + evaluation da solução inicial
initial_feas = feasibility_metrics(
    assignments=initial_assignments_seq,
    df_rooms=df_rooms,
    df_surgeons=df_surgeons,
    patients=df_patients,
    C_PER_SHIFT=C_PER_SHIFT
)

initial_eval = evaluate_schedule(
    assignments=initial_assignments_seq,
    patients=df_patients,
    rooms_free=initial_rooms_free,
    excess_block_min=initial_feas["excess_block_min"]
)

    
# =========================================================
#        ITERATED LOCAL SEARCH (SIMPLIFICADO)
# =========================================================

N_ILS_ITER = 30
MAX_SWAP_OUT = 2
MAX_SWAP_IN  = 2

# 1) SOLUÇÃO CORRENTE = SOLUÇÃO INICIAL (DEPOIS DO HEURÍSTICO)
current_assignments = df_assignments.copy()

# Função auxiliar: corre TODA a pipeline do teu código:
def full_evaluation(assignments):
    # 1) Merge com pacientes
    enriched = assignments.merge(
        df_patients[["patient_id","duration","priority","waiting"]],
        on="patient_id",
        how="left"
    )

    # 2) Sequenciamento (resolve overlap + tira cirurgias problemáticas)
    enriched = sequence_global_by_surgeon(
        enriched,
        C_PER_SHIFT=C_PER_SHIFT,
        CLEANUP=CLEANUP,
        TOLERANCE=TOLERANCE,
        ROOM_CHANGE_TIME=ROOM_CHANGE_TIME
    )

    seq = enriched[enriched["scheduled_by_seq"]==1].copy()

    # 3) Recalcular rooms_free
    rooms_free = build_room_free_from_assignments(seq, df_rooms, C_PER_SHIFT)

    # 4) Feasibility
    feas = feasibility_metrics(
        assignments=seq,
        df_rooms=df_rooms,
        df_surgeons=df_surgeons,
        patients=df_patients,
        C_PER_SHIFT=C_PER_SHIFT
    )

    # 5) Evaluation (A TUA FUNÇÃO)
    ev = evaluate_schedule(
        assignments=seq,
        patients=df_patients,
        rooms_free=rooms_free,
        excess_block_min=feas["excess_block_min"]
    )

    return ev["score"], seq, rooms_free, feas, enriched


# Avaliar solução inicial
current_score, current_seq, current_rooms_free, current_feas, _ = full_evaluation(current_assignments)

best_score = current_score
best_assignments = current_assignments.copy()
best_seq = current_seq.copy()
best_rooms_free = current_rooms_free.copy()
best_feas = current_feas  

print("\nILS START")
print("Initial score:", current_score)

for it in range(N_ILS_ITER):

    # Gerar vizinho (swap i↔j)
    neighbor, ids_out, ids_in = generate_neighbor_swap(
        current_assignments,
        df_patients,
        df_rooms,
        df_surgeons,
        C_PER_SHIFT,
        max_swap_out=MAX_SWAP_OUT,
        max_swap_in=MAX_SWAP_IN
    )

    neigh_score, neigh_seq, neigh_rooms_free, neigh_feas, _ = full_evaluation(neighbor)

    # Aceitar se SCORE melhorar
    if neigh_score > current_score:
        current_assignments = neighbor.copy()
        current_score = neigh_score

        # Atualizar melhor global
        if neigh_score > best_score:
            best_score = neigh_score
            best_assignments = neighbor.copy()
            best_seq = neigh_seq.copy()
            best_rooms_free = neigh_rooms_free.copy()
            best_feas = neigh_feas 

        print(f"[Iter {it}] Improved score to {current_score:.4f} | "
              f"removed={ids_out} | added={ids_in}")
    else:
       None
       

# ============================================================
#     CONSTRUIR A SOLUÇÃO FINAL (COM SEQUENCIAMENTO)
# ============================================================

# 1) Merge com pacientes
best_assignments_enriched = best_assignments.merge(
    df_patients[["patient_id","duration","priority","waiting"]],
    on="patient_id",
    how="left"
)

# 2) Sequenciamento final (trata overlaps e scheduled_by_seq)
best_assignments_enriched = sequence_global_by_surgeon(
    best_assignments_enriched,
    C_PER_SHIFT=C_PER_SHIFT,
    CLEANUP=CLEANUP,
    TOLERANCE=TOLERANCE,
    ROOM_CHANGE_TIME=ROOM_CHANGE_TIME
)

# 3) Cirurgias que ficam na solução final
best_assignments_seq = best_assignments_enriched[
    best_assignments_enriched["scheduled_by_seq"] == 1
].sort_values(["day","shift","room","seq_in_block"]).reset_index(drop=True)

# 4) Cirurgias removidas pelo sequenciador
best_assignments_dropped = best_assignments_enriched[
    best_assignments_enriched["scheduled_by_seq"] == 0
].sort_values(["day","shift","room"]).reset_index(drop=True)

# 5) Recalcular rooms_free (já tens best_rooms_free mas confirmamos)
best_rooms_free = build_room_free_from_assignments(
    assignments=best_assignments_seq,
    df_rooms=df_rooms,
    C_PER_SHIFT=C_PER_SHIFT
)

# 6) Recalcular surgeons_free
best_surgeon_free = build_surgeon_free_from_assignments(
    assignments=best_assignments_seq,
    df_surgeons=df_surgeons,
    C_PER_SHIFT=C_PER_SHIFT
)

          
# A PARTIR DE AGORA, USAMOS A MELHOR SOLUÇÃO DO ILS
assignments_enriched = best_assignments_enriched.copy()
assignments_seq_view = best_assignments_seq.copy()
df_room_free = best_rooms_free.copy()
df_surgeon_free = best_surgeon_free.copy()


# --------------------------------------------
# RECOMPUTE room/surgeon usage based on sequenced assignments
# --------------------------------------------
#df_room_free = build_room_free_from_assignments(
#    assignments=assignments_seq_view,
#    df_rooms=df_rooms,
#    C_PER_SHIFT=C_PER_SHIFT
#)

#df_surgeon_free = build_surgeon_free_from_assignments(
#    assignments=assignments_seq_view,
#    df_surgeons=df_surgeons,
 #   C_PER_SHIFT=C_PER_SHIFT
#)


# ============================================================
# EXPORT PACK — build all relevant tables and write to Excel
# ============================================================
from datetime import datetime

# ---------- 0) helpers ----------
ts = datetime.now().strftime("%Y%m%d_%H%M")
xlsx_path = f"or_schedule_export_{ts}.xlsx"

# ---------- 1) Inputs (nice tabular forms) ----------
inputs_patients = df_patients.sort_values("patient_id").copy()
inputs_rooms = df_rooms.sort_values(["room", "day", "shift"]).copy()
inputs_surgeons = df_surgeons.sort_values(["surgeon_id", "day", "shift"]).copy()

rooms_av_matrix = inputs_rooms.pivot_table(
    index=["room", "day"], columns="shift", values="available", aggfunc="first"
).rename(columns={1: "AM", 2: "PM"}).reset_index()

surgeons_av_matrix = inputs_surgeons.pivot_table(
    index=["surgeon_id", "day"], columns="shift", values="available", aggfunc="first"
).rename(columns={1: "AM", 2: "PM"}).reset_index()



# ---------- 3) Capacity snapshots (final) ----------
rooms_free = df_room_free.copy()
surgeons_free = df_surgeon_free.copy()



# ---------- 4) KPIs / summaries ----------
kpi_day_rooms = rooms_free.groupby("day", as_index=False).agg(
    open_blocks=("available", "sum"),
    cap_min=("cap_min", "sum"),
    used_min=("used_min", "sum"),
    free_min=("free_min", "sum")
)
kpi_day_rooms["utilization"] = kpi_day_rooms.apply(
    lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0.0, axis=1
)

kpi_room = rooms_free.groupby("room", as_index=False).agg(
    open_blocks=("available", "sum"),
    cap_min=("cap_min", "sum"),
    used_min=("used_min", "sum"),
    free_min=("free_min", "sum")
)
kpi_room["utilization"] = kpi_room.apply(
    lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0.0, axis=1
)

kpi_surgeon = surgeons_free.groupby("surgeon_id", as_index=False).agg(
    open_blocks=("available", "sum"),
    cap_min=("cap_min", "sum"),
    used_min=("used_min", "sum"),
    free_min=("free_min", "sum")
)
kpi_surgeon["utilization"] = kpi_surgeon.apply(
    lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0.0, axis=1
)

total_cap = rooms_free["cap_min"].sum()
total_used = rooms_free["used_min"].sum()
total_free = rooms_free["free_min"].sum()
global_util = (total_used / total_cap) if total_cap > 0 else 0.0

n_assigned = len(assignments_seq_view)
assigned_ids = set(assignments_seq_view["patient_id"].unique())
n_unassigned = len(df_patients) - n_assigned

kpi_global = pd.DataFrame([{
    "total_capacity_min": total_cap,
    "total_used_min": total_used,
    "total_free_min": total_free,
    "global_utilization": global_util,
    "assigned_patients": n_assigned,
    "unassigned_patients": n_unassigned
}])




# ---------- 5) Unassigned patients ----------
unassigned_patients = df_patients[
    ~df_patients["patient_id"].isin(assigned_ids)
].sort_values("patient_id").copy()

if len(unassigned_patients):
    blocks_open = df_rooms[df_rooms["available"] == 1][
        ["room", "day", "shift"]
    ].drop_duplicates()
    surg_open   = df_surgeons[df_surgeons["available"] == 1][
        ["surgeon_id", "day", "shift"]
    ].drop_duplicates()
    pmini_u = unassigned_patients[[
        "patient_id", "surgeon_id", "duration"
    ]].copy()
    ptime_u = pmini_u.merge(surg_open, on="surgeon_id", how="inner")
    pblocks_u = ptime_u.merge(blocks_open, on=["day", "shift"], how="inner")
    pblocks_u["fits_shift"] = (pblocks_u["duration"] + CLEANUP) <= C_PER_SHIFT
    pblocks_u = pblocks_u[pblocks_u["fits_shift"]]
    feas_u = (
        pblocks_u.groupby("patient_id", as_index=False)
                 .agg(feasible_blocks=("room", "count"))
    )
    unassigned_patients = unassigned_patients.merge(
        feas_u, on="patient_id", how="left"
    ).fillna({"feasible_blocks": 0})

# ---------- 6) Final block state (for audit) ----------
cases_per_block = (
    assignments_seq_view.groupby(["room", "day", "shift"], as_index=False)
                       .size()
                       .rename(columns={"size": "n_cases"})
)
final_blocks = rooms_free.merge(
    cases_per_block, on=["room", "day", "shift"], how="left"
).fillna({"n_cases": 0})

# ---------- 7) Write everything to Excel ----------
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    # Inputs
    inputs_patients.to_excel(writer, sheet_name="Inputs_Patients", index=False)
    inputs_rooms.to_excel(writer, sheet_name="Inputs_Rooms", index=False)
    inputs_surgeons.to_excel(writer, sheet_name="Inputs_Surgeons", index=False)
    rooms_av_matrix.to_excel(writer, sheet_name="Rooms_Availability_Matrix", index=False)
    surgeons_av_matrix.to_excel(writer, sheet_name="Surgeons_Availability_Matrix", index=False)

    # ============ RESULTS – SOLUÇÃO INICIAL (heurístico) ============
    initial_assignments_enriched.to_excel(
        writer, sheet_name="Assignments_Initial", index=False
    )
    initial_assignments_seq.to_excel(
        writer, sheet_name="Assignments_Sequenced_Initial", index=False
    )
    initial_rooms_free.to_excel(
        writer, sheet_name="Rooms_Free_Initial", index=False
    )
    initial_surgeons_free.to_excel(
        writer, sheet_name="Surgeons_Free_Initial", index=False
    )

    # ============ RESULTS – SOLUÇÃO FINAL (ILS) ============
    assignments_enriched.to_excel(
        writer, sheet_name="Assignments_ILS", index=False
    )
    assignments_seq_view.to_excel(
        writer, sheet_name="Assignments_Sequenced_ILS", index=False
    )
    final_blocks.to_excel(writer, sheet_name="Blocks_FinalState_ILS", index=False)
    rooms_free.to_excel(writer, sheet_name="Rooms_Free_ILS", index=False)
    surgeons_free.to_excel(writer, sheet_name="Surgeons_Free_ILS", index=False)

    # KPIs (já estão calculados para a solução final)
    kpi_global.to_excel(writer, sheet_name="KPI_Global_ILS", index=False)
    kpi_day_rooms.to_excel(writer, sheet_name="KPI_PerDay_ILS", index=False)
    kpi_room.to_excel(writer, sheet_name="KPI_PerRoom_ILS", index=False)
    kpi_surgeon.to_excel(writer, sheet_name="KPI_PerSurgeon_ILS", index=False)

    # Unassigned (da solução final)
    unassigned_patients.to_excel(writer, sheet_name="Unassigned_ILS", index=False)


#print(f"\nExcel exported → {xlsx_path}")


# ---------- 8) TEXT-BASED SCHEDULE (formato tipo imagem) ----------

print("\n==================== FINAL SCHEDULE ====================\n")

if len(assignments_seq_view) == 0:
    print("(No assignments found — nothing to display.)")
else:
    # Usamos a solução FINAL sequenciada
    INCLUDE_CLEANUP_IN_TIMELINE = True  # só afeta a interpretação do end, se quiseres

    # ordenar por dia, turno, sala e tempo de início real
    assignments_sorted = assignments_seq_view.sort_values(
        ["day", "shift", "room", "start_min"]
    )

    for (r, d, sh), group in assignments_sorted.groupby(
        ["room", "day", "shift"], sort=True
    ):
        # Header do bloco (0-based, como na imagem original)
        print(f"\nB_{int(r)-1}_{int(d)-1}_{int(sh)-1}:")

        for _, row in group.iterrows():
            pid = int(row["patient_id"])
            sid = int(row["surgeon_id"])
            dur = int(row["duration"])

            start = int(row["start_min"])

            end = int(row["end_min"])  
            

            print(f"  (p={pid}, s={sid}, dur={dur}, start={start}, end={end})")

    print("\n========================================================\n")

# --------------------------------------------
# RUN FEASIBILITY + EVALUATION ON initial SOLUTION 
# --------------------------------------------
feas = feasibility_metrics(
    assignments=assignments_seq_view,   
    df_rooms=df_rooms,
    df_surgeons=df_surgeons,
    patients=df_patients,
    C_PER_SHIFT=C_PER_SHIFT
)


# ------- SCORE INICIAL -------

initial_eval = evaluate_schedule(
    assignments=assignments_seq_view,
    patients =df_patients,
    rooms_free=df_room_free,
    excess_block_min=feas["excess_block_min"],
)

# KPIs com a solução final
best_eval = evaluate_schedule(
    assignments=best_assignments_seq,
    patients=df_patients,
    rooms_free=best_rooms_free,
    excess_block_min=best_feas["excess_block_min"]
)



print("\n==================== INITIAL KPIs =====================\n")

print(">>> Evaluation KPIs (Initial)")
print(f" Score:              {initial_eval['score']:.4f}")
print(f" Ratio Scheduled:    {initial_eval['ratio_scheduled']:.3f}")
print(f" Room Utilization:   {initial_eval['util_rooms']:.3f}")
print(f" Priority Rate:      {initial_eval['prio_rate']:.3f}")

# Average waiting time of scheduled patients (initial)
if len(initial_assignments_seq) > 0:
    initial_avg_wait = initial_assignments_seq["waiting"].mean()
else:
    initial_avg_wait = 0

print(f" Average waiting time (days): {initial_avg_wait:.2f}")

print("\n>>> Feasibility KPIs (Initial)")
print(f" Unassigned patients:        {initial_feas['n_unassigned']}")
print(f" Excess minutes in blocks:   {initial_feas['excess_block_min']}")
print(f" Excess surgeon minutes:     {initial_feas['excess_surgeon_min']}")
print(f" Block availability viol:    {initial_feas['block_unavailable_viol']}")
print(f" Surgeon availability viol:  {initial_feas['surg_unavailable_viol']}")
print(f" Feasibility score:          {initial_feas['feasibility_score']}")

# Global overview
print("\n>>> Global Overview (Initial)")
print(f" Total scheduled patients = {len(initial_assignments_seq)}")
print(f" Total capacity minutes   = {initial_rooms_free['cap_min'].sum()}")
print(f" Total used minutes       = {initial_rooms_free['used_min'].sum()}")
print(f" Total free minutes       = {initial_rooms_free['free_min'].sum()}")
print(f" Global utilization       = {initial_rooms_free['used_min'].sum() / initial_rooms_free['cap_min'].sum():.3f}")

print("\n=======================================================\n")
