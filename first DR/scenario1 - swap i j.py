# -*- coding: utf-8 -*-
"""
Operating Room Scheduling — Step 1 + Step 2 (Iterative Dispatching Rule)
Author: Joana
"""

from pathlib import Path
import re
import ast
import pandas as pd
import random

# ------------------------------
# PARAMETERS
# ------------------------------
DATA_FILE = "instance_c1_30.dat"

C_PER_SHIFT = 360   # minutes per shift (6h * 60)
CLEANUP = 17        # cleaning time

ALPHA1 = 0.10  # priority
ALPHA2 = 0.10  # waited days
ALPHA3 = 0.10 # deadline closeness
ALPHA4 = 0.70  # feasible blocks

ALPHA5 = 1/3  
ALPHA6 = 1/3
ALPHA7 = 1/3 


TOLERANCE = 15  # NEW- tolerance minutes to pass the capacity of the shift


# ------------------------------
# FILE READING UTILITIES
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
    """Return feasible (room, day, shift) given current capacity & surgeon load."""
    sid = int(patient_row["surgeon_id"])
    need = int(patient_row["duration"]) + CLEANUP

    # surgeon available (day, shift)
    surg_ok = df_surgeons[(df_surgeons["surgeon_id"] == sid) &
                          (df_surgeons["available"] == 1)][["day", "shift"]]

    # rooms open with enough capacity
    cap_ok = df_capacity[(df_capacity["available"] == 1) &
                         (df_capacity["free_min"] >= need)][["room", "day", "shift", "free_min"]]

    cand = surg_ok.merge(cap_ok, on=["day", "shift"], how="inner")

    # surgeon load within shift capacity
    surg_load = df_surgeon_load[df_surgeon_load["surgeon_id"] == sid][["day", "shift", "used_min"]]
    cand = cand.merge(surg_load, on=["day", "shift"], how="left").fillna({"used_min": 0})
    cand = cand[(cand["used_min"] + need) <= C_PER_SHIFT + TOLERANCE]

    # continuity flag (already operating in same block)
    if len(df_assignments) > 0:
        cont = df_assignments[df_assignments["surgeon_id"] == sid][["room", "day", "shift"]].copy()
        cont["continuity"] = 1
        cand = cand.merge(cont, on=["room", "day", "shift"], how="left")
    else:
        cand["continuity"] = 0

    cand["continuity"] = cand["continuity"].fillna(0).astype(int)

    return cand


def score_block_for_patient(cand_df, patient_row, n_days):
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
    """Update capacity, surgeon-load, and record assignment with iteration order."""
    pid = int(patient_row["patient_id"])
    sid = int(patient_row["surgeon_id"])
    dur_need = int(patient_row["duration"]) + CLEANUP
    r, d, sh = int(best_row["room"]), int(best_row["day"]), int(best_row["shift"])

    # update capacity
    idx = (df_capacity["room"] == r) & (df_capacity["day"] == d) & (df_capacity["shift"] == sh)
    df_capacity.loc[idx, "free_min"] -= dur_need

    # update surgeon load
    idx_s = (df_surgeon_load["surgeon_id"] == sid) & (df_surgeon_load["day"] == d) & (df_surgeon_load["shift"] == sh)
    df_surgeon_load.loc[idx_s, "used_min"] += dur_need

    # record assignment (store iteration and optional scores)
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
    return 3 if p==3 else (15 if p == 2 else (90 if p == 1 else 270))

def deadline_term(priority, waited):
    lim = deadline_limit_from_priority(priority)
    if lim is None: return 0.0
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

     # 6) violações de prazo clínico (waiting days > limite da prioridade)
    pats = patients.copy()
    pats["deadline_limit"] = pats["priority"].apply(deadline_limit_from_priority)
    pats["overdue_days"] = (pats["waiting"] - pats["deadline_limit"]).clip(lower=0)

    total_overdue_days = int(pats["overdue_days"].sum())
    n_overdue_patients = int((pats["overdue_days"] > 0).sum())

    # score total de infeasibilidade (minimizar; 0 => solução "ideal")
    feasibility_score = (
          block_unavailable_viol
        + surg_unavailable_viol
        + excess_block_min
        + excess_surgeon_min
        + total_overdue_days      # penaliza atraso em dias
        + n_overdue_patients      # penaliza nº de doentes em atraso
    )

    return {
        "n_unassigned": n_unassigned,
        "block_unavailable_viol": block_unavailable_viol,
        "surg_unavailable_viol": surg_unavailable_viol,
        "excess_block_min": excess_block_min,
        "excess_surgeon_min": excess_surgeon_min,
        "total_overdue_days": total_overdue_days,
        "n_overdue_patients": n_overdue_patients,
        "feasibility_score": feasibility_score,
        "rooms_cap_join": rooms_join
    }

def evaluate_schedule(assignments, patients, rooms_free, excess_block_min,
                      weights=(0.6, 0.1, 0.25, 0.05)):
    w1, w2, w3, w4 = weights
    total_patients = len(patients)
    ratio_scheduled = (len(assignments) / total_patients) if total_patients else 0.0

    util_rooms = float(
        rooms_free.loc[rooms_free["cap_min"] > 0, "utilization"].mean()
    ) if len(rooms_free) else 0.0

    if len(assignments):
        merged = assignments.copy()  
        # ------------------------------
        # 1) PRIORITY TERM (normalizado)
        # ------------------------------
        pmax = float(patients["priority"].max())
        prio_rate = float(merged["priority"].mean() / pmax) if pmax > 0 else 0.0

        # ------------------------------
        # 2) WAITING TERM + DEADLINE PENALTY
        # ------------------------------
        wmax = float(patients["waiting"].max())

        # base: esperar mais = score maior
        if wmax > 0:
            base_wait_term = 1.0 - (float(merged["waiting"].mean()) / wmax)
        else:
            base_wait_term = 1.0

        # limite por prioridade
        merged["deadline_limit"] = merged["priority"].apply(deadline_limit_from_priority)

        # atraso face ao limite clínico
        merged["overdue_days"] = (merged["waiting"] - merged["deadline_limit"]).clip(lower=0)
        merged["overdue_frac"] = merged["overdue_days"] / merged["deadline_limit"]

        avg_overdue_frac = float(merged["overdue_frac"].mean())

        # termo final de waiting
        norm_wait_term = max(0.0, base_wait_term - avg_overdue_frac)

    else:
        prio_rate = 0.0
        norm_wait_term = 0.0

    # ------------------------------
    # 3) SCORE FINAL
    # ------------------------------
    score = (
        w1 * ratio_scheduled +
        w2 * util_rooms +
        w3 * prio_rate +
        w4 * norm_wait_term -
        0.001 * excess_block_min
    )

    return {
        "score": float(score),
        "ratio_scheduled": float(ratio_scheduled),
        "util_rooms": float(util_rooms),
        "prio_rate": float(prio_rate),
        "norm_wait_term": float(norm_wait_term)
    }





def candidate_blocks_for_patient_in_solution(assignments, patient_row,
                                             df_rooms, df_surgeons,
                                             C_PER_SHIFT):
    """
    Devolve blocos (room, day, shift, free_min) onde o doente poderia ficar,
    dado o estado atual da solução assignments.
    Cenário 1: se o cirurgião já está nesse (day,shift), deve ficar na mesma sala.
    """
    sid = int(patient_row["surgeon_id"])
    need = int(patient_row["duration"]) + CLEANUP

    # 1) disponibilidade do cirurgião
    surg_ok = df_surgeons[
        (df_surgeons["surgeon_id"] == sid) &
        (df_surgeons["available"] == 1)
    ][["day","shift"]]

    # 2) capacidade por bloco (a partir do assignments atual)
    rooms_base = df_rooms[["room","day","shift","available"]].copy()
    rooms_base["cap_min"] = rooms_base["available"] * C_PER_SHIFT

    if len(assignments):
        used_by_block = (assignments.groupby(["room","day","shift"], as_index=False)
                         .agg(used_min=("used_min","sum")))
    else:
        used_by_block = rooms_base[["room","day","shift"]].copy()
        used_by_block["used_min"] = 0

    rooms_join = rooms_base.merge(
        used_by_block,
        on=["room","day","shift"],
        how="left"
    ).fillna({"used_min": 0})
    rooms_join["free_min"] = (rooms_join["cap_min"] - rooms_join["used_min"]).clip(lower=0)

    cap_ok = rooms_join[
        (rooms_join["available"] == 1) &
        (rooms_join["free_min"] >= need)
    ][["room","day","shift","free_min"]]

    cand = surg_ok.merge(cap_ok, on=["day","shift"], how="inner")

    # 3) carga do cirurgião por (day,shift)
    if len(assignments):
        sload = (assignments[assignments["surgeon_id"] == sid]
                 .groupby(["day","shift"], as_index=False)
                 .agg(used_min=("used_min","sum")))
    else:
        sload = pd.DataFrame(columns=["day","shift","used_min"])

    cand = cand.merge(sload, on=["day","shift"], how="left").fillna({"used_min": 0})
    cand = cand[(cand["used_min"] + need) <= C_PER_SHIFT]

    # 4) Cenário 1: se já opera nesse (day,shift), deve ficar na mesma sala
    if len(assignments):
        locks = (assignments[assignments["surgeon_id"] == sid]
                 [["day","shift","room"]].drop_duplicates()
                 .rename(columns={"room":"room_locked"}))
        cand = cand.merge(locks, on=["day","shift"], how="left")
        cand = cand[
            (cand["room_locked"].isna()) |
            (cand["room"] == cand["room_locked"])
        ]
        cand = cand.drop(columns=["room_locked"])

    return cand


def generate_neighbor_swap(current_assignments,
                           df_patients,
                           df_rooms,
                           df_surgeons,
                           C_PER_SHIFT,
                           max_swap_out=2,
                           max_swap_in=2):

    assignments = current_assignments.copy()

    # pacientes atualmente agendados
    scheduled_ids = assignments["patient_id"].unique().tolist()
    if len(scheduled_ids) == 0:
        return assignments, [], []

    # pacientes não agendados
    all_ids = df_patients["patient_id"].unique().tolist()
    unassigned_ids = [pid for pid in all_ids if pid not in scheduled_ids]

    if len(unassigned_ids) == 0:
        return assignments, [], []

    # ---------- 1) REMOVER i pacientes (até max_swap_out) ----------
    k_out = min(max_swap_out, len(scheduled_ids))
    ids_out = random.sample(scheduled_ids, k_out)

    assignments = assignments[~assignments["patient_id"].isin(ids_out)].copy()

    # ---------- 2) ADICIONAR j pacientes (até max_swap_in) ----------
    random.shuffle(unassigned_ids)
    ids_in_candidates = unassigned_ids[:max_swap_in]

    ids_in_effective = []

    for pid in ids_in_candidates:
        prow = df_patients[df_patients["patient_id"] == pid]
        if prow.empty:
            continue
        prow = prow.iloc[0]

        # blocos viáveis dado o assignments ATUAL
        cand_blocks = candidate_blocks_for_patient_in_solution(
            assignments,
            prow,
            df_rooms,
            df_surgeons,
            C_PER_SHIFT
        )

        if cand_blocks.empty:
            continue

        # escolhe um bloco simples (p.ex. mais cedo no tempo)
        chosen = cand_blocks.sort_values(
            ["day", "shift", "room"]
        ).iloc[0]

        dur = int(prow["duration"])
        sid = int(prow["surgeon_id"])
        need = dur + CLEANUP

        new_row = {
            "patient_id": int(pid),
            "room": int(chosen["room"]),
            "day": int(chosen["day"]),
            "shift": int(chosen["shift"]),
            "used_min": need,
            "surgeon_id": sid,
            "iteration": -1,      # marca que veio da LS/ILS
            "W_patient": None,
            "W_block": None,
        }

        new_row_df = pd.DataFrame([new_row])
        
        if assignments.empty:
            assignments = new_row_df.copy()
        else:
            assignments = pd.concat([assignments, new_row_df], ignore_index=True)
                
        

        ids_in_effective.append(int(pid))

    return assignments, ids_out, ids_in_effective

# ------------------------------
# INITIAL PLANNING STATE
# ------------------------------
df_capacity = df_rooms.copy()
df_capacity["free_min"] = df_capacity["available"].apply(lambda a: C_PER_SHIFT if a == 1 else 0)

df_assignments = pd.DataFrame(columns=["patient_id", "room", "day", "shift", "used_min", "surgeon_id", "iteration", "W_patient", "W_block"])

df_surgeon_load = df_surgeons[["surgeon_id", "day", "shift"]].drop_duplicates().assign(used_min=0)


# ------------------------------
# ITERATIVE LOOP: Step 1 → Step 2 → commit → repeat
# ------------------------------
remaining = df_patients.copy()
iteration = 0

while True:
    iteration += 1

    # ---- Step 1: dynamic feasible blocks per patient (uses current capacity & surgeon load) ----
    df_surg_open = df_surgeons[df_surgeons["available"] == 1][["surgeon_id", "day", "shift"]].drop_duplicates()
    df_cap_open  = df_capacity[df_capacity["available"] == 1][["room", "day", "shift", "free_min"]].drop_duplicates()
    df_sload     = df_surgeon_load[["surgeon_id","day","shift","used_min"]].drop_duplicates()
    
    df_pmini = remaining[["patient_id","surgeon_id","duration","priority","waiting"]].copy()
    df_pmini["need"] = df_pmini["duration"] + CLEANUP
    
    # (surgeon availability)
    df_p_time = df_pmini.merge(df_surg_open, on="surgeon_id", how="inner")
    
    # (join current room capacity)
    df_p_cap = df_p_time.merge(df_cap_open, on=["day","shift"], how="inner")
    
    # (current surgeon load per (day,shift))
    df_p_cap = df_p_cap.merge(df_sload, on=["surgeon_id","day","shift"], how="left").fillna({"used_min":0})
    
    # keep only blocks that can host the case now
    df_p_blocks = df_p_cap[(df_p_cap["free_min"] >= df_p_cap["need"]) &
                           ((df_p_cap["used_min"] + df_p_cap["need"]) <= C_PER_SHIFT + TOLERANCE)] # NEW
    # to see where was the overflow
    df_p_blocks["overflow_min"] = (
    (df_p_cap["used_min"] + df_p_cap["need"]) - C_PER_SHIFT).clip(lower=0) 
    #overflow_min = 0 if is within the limits
    #overflow_min = 2 if passes 2 minutes...

    
    # count feasible blocks per patient
    df_feas_count = (df_p_blocks.groupby("patient_id", as_index=False)
                                .agg(feasible_blocks=("room","count")))
    
    step1 = df_pmini.merge(df_feas_count, on="patient_id", how="left").fillna({"feasible_blocks":0})


    # ---- Step 1 scoring
    Pmax = max(step1["priority"].max(), 1)
    Wmax = max(step1["waiting"].max(), 1)
    step1["term_priority"] = step1["priority"] / Pmax
    step1["term_waiting"]  = step1["waiting"] / Wmax
    step1["term_deadline"] = step1.apply(lambda r: deadline_term(r["priority"], r["waiting"]), axis=1)
    step1["term_scarcity"] = 1.0 / (1.0 + step1["feasible_blocks"])
    step1["W_patient"] = (
          ALPHA1 * step1["term_priority"]
        + ALPHA2 * step1["term_waiting"]
        + ALPHA3 * step1["term_deadline"]
        + ALPHA4 * step1["term_scarcity"]
    )

    # stop if no feasible patients remain
    if step1["feasible_blocks"].fillna(0).max() == 0:
        print("\nNo more schedulable patients under Step-1 filters.")
        break

    # pick next patient (highest W_patient)
    patient_row = step1.sort_values("W_patient", ascending=False).iloc[0]

    # ---- Step 2: feasible blocks with current state
    cand_blocks = feasible_blocks_step2(patient_row)
    if cand_blocks.empty:
        remaining = remaining[remaining["patient_id"] != patient_row["patient_id"]]
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


    # remove scheduled patient
    remaining = remaining[remaining["patient_id"] != patient_row["patient_id"]]

    # progress log
 
feas_init = feasibility_metrics(df_assignments, df_rooms, df_surgeons, df_patients, C_PER_SHIFT)
score_init = evaluate_schedule(df_assignments, df_patients, feas_init["rooms_join"])

assigned_ids = set(df_assignments["patient_id"])
unassigned_patients = df_patients[~df_patients["patient_id"].isin(assigned_ids)].copy()
print(unassigned_patients["waiting"].mean())
print("aquiii")


# Rooms: base capacity
rooms_base = df_rooms[["room", "day", "shift", "available"]].copy()
rooms_base["cap_min"] = rooms_base["available"] * C_PER_SHIFT

if len(df_assignments):
    used_by_block = (
        df_assignments
        .groupby(["room", "day", "shift"], as_index=False)
        .agg(used_min=("used_min", "sum"))
    )
else:
    used_by_block = rooms_base[["room", "day", "shift"]].copy()
    used_by_block["used_min"] = 0

rooms_free = rooms_base.merge(
    used_by_block, on=["room", "day", "shift"], how="left"
).fillna({"used_min": 0})

rooms_free["free_min"] = (rooms_free["cap_min"] - rooms_free["used_min"]).clip(lower=0)
rooms_free["utilization"] = rooms_free.apply(
    lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0.0,
    axis=1
)
rooms_free = rooms_free.sort_values(["room", "day", "shift"]).reset_index(drop=True)
print(rooms_free)
print(rooms_free[rooms_free["utilization"] != 0]["utilization"].mean())
print("utilizaçaooo")

print(df_assignments)
# ============================================
# LOCAL SEARCH / ILS com vizinho swap i→j
# ============================================

N_ILS_ITER = 30       # nº de iterações da LS/ILS (ajusta como quiseres)
MAX_SWAP_OUT = 2      # i: nº máximo de pacientes a remover
MAX_SWAP_IN  = 2      # j: nº máximo de pacientes a tentar adicionar

# solução corrente começa na solução construtiva
current_assignments = df_assignments.copy()

# score da solução inicial (APENAS com evaluate_schedule)
feas_init_ls = feasibility_metrics(
    current_assignments, df_rooms, df_surgeons, df_patients, C_PER_SHIFT
)
current_score = evaluate_schedule(
    current_assignments,
    df_patients,
    feas_init_ls["rooms_join"]
)

best_assignments = current_assignments.copy()
best_score = current_score
best_feas = feas_init_ls  # só para ter guardado, se quiseres ver depois

print("\nILS START")
print(f"Initial LS score = {current_score:.4f}, feas = {feas_init_ls['feasibility_score']}")

for it in range(N_ILS_ITER):
    # gerar vizinho com swap i->j
    neighbor, ids_out, ids_in = generate_neighbor_swap(
        current_assignments,
        df_patients,
        df_rooms,
        df_surgeons,
        C_PER_SHIFT,
        max_swap_out=MAX_SWAP_OUT,
        max_swap_in=MAX_SWAP_IN
    )

    # avaliar vizinho
    feas_neigh = feasibility_metrics(
        neighbor, df_rooms, df_surgeons, df_patients, C_PER_SHIFT
    )
    score_neigh = evaluate_schedule(
        neighbor,
        df_patients,
        feas_neigh["rooms_join"]
    )

    # se quiseres recusar sempre soluções inviáveis, podes pôr:
    # if feas_neigh["feasibility_score"] > 0:
    #     continue

    if score_neigh > current_score:
        current_assignments = neighbor.copy()
        current_score = score_neigh

        if score_neigh > best_score:
            best_assignments = neighbor.copy()
            best_score = score_neigh
            best_feas = feas_neigh

        print(f"[Iter {it}] improved to {current_score:.4f} | "
              f"removed={ids_out} | added={ids_in}")


print(f"Score final = {best_score:.4f}, feasibility_score = {best_feas['feasibility_score']}")

# --------------------------------------------
# REBUILD room & surgeon usage from best_assignments
# --------------------------------------------

# Rooms: base capacity
rooms_base = df_rooms[["room", "day", "shift", "available"]].copy()
rooms_base["cap_min"] = rooms_base["available"] * C_PER_SHIFT

if len(best_assignments):
    used_by_block = (
        best_assignments
        .groupby(["room", "day", "shift"], as_index=False)
        .agg(used_min=("used_min", "sum"))
    )
else:
    used_by_block = rooms_base[["room", "day", "shift"]].copy()
    used_by_block["used_min"] = 0

rooms_free = rooms_base.merge(
    used_by_block, on=["room", "day", "shift"], how="left"
).fillna({"used_min": 0})

rooms_free["free_min"] = (rooms_free["cap_min"] - rooms_free["used_min"]).clip(lower=0)
rooms_free["utilization"] = rooms_free.apply(
    lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0.0,
    axis=1
)
rooms_free = rooms_free.sort_values(["room", "day", "shift"]).reset_index(drop=True)




# Surgeons: base availability
surg_base = df_surgeons[["surgeon_id", "day", "shift", "available"]].drop_duplicates()

if len(best_assignments):
    sload = (
        best_assignments
        .groupby(["surgeon_id", "day", "shift"], as_index=False)
        .agg(used_min=("used_min", "sum"))
    )
else:
    sload = surg_base[["surgeon_id", "day", "shift"]].copy()
    sload["used_min"] = 0

surgeons_free = surg_base.merge(
    sload, on=["surgeon_id", "day", "shift"], how="left"
).fillna({"used_min": 0})

surgeons_free["cap_min"] = surgeons_free["available"] * C_PER_SHIFT
surgeons_free["free_min"] = (surgeons_free["cap_min"] - surgeons_free["used_min"]).clip(lower=0)
surgeons_free["utilization"] = surgeons_free.apply(
    lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0.0,
    axis=1
)
surgeons_free = surgeons_free.sort_values(["surgeon_id", "day", "shift"]).reset_index(drop=True)




# ============================================================
# EXPORT PACK — build all relevant tables and write to Excel
# ============================================================
import pandas as pd
from datetime import datetime

# ---------- 0) helpers ----------
ts = datetime.now().strftime("%Y%m%d_%H%M")
xlsx_path = f"or_schedule_export_{ts}.xlsx"

# ---------- 1) Inputs (nice tabular forms) ----------
# Patients input table (as read)
inputs_patients = df_patients.sort_values("patient_id").copy()

# Rooms availability (room/day/shift) as 0/1
inputs_rooms = df_rooms.sort_values(["room", "day", "shift"]).copy()

# Surgeons availability (surgeon/day/shift) as 0/1
inputs_surgeons = df_surgeons.sort_values(["surgeon_id", "day", "shift"]).copy()

# Optional: matrix-style pivots for human reading
rooms_av_matrix = inputs_rooms.pivot_table(
    index=["room", "day"], columns="shift", values="available", aggfunc="first"
).rename(columns={1:"AM", 2:"PM"}).reset_index()

surgeons_av_matrix = inputs_surgeons.pivot_table(
    index=["surgeon_id", "day"], columns="shift", values="available", aggfunc="first"
).rename(columns={1:"AM", 2:"PM"}).reset_index()

# ---------- 2) Assignments enriched ----------
# Join extra patient info to assignments
assignments_enriched = best_assignments.merge(
    df_patients[["patient_id", "duration", "priority", "waiting"]],
    on="patient_id",
    how="left"
).sort_values("iteration")

# Add a simple sequence number per (room,day,shift)
assignments_enriched["seq_in_block"] = (
    assignments_enriched.groupby(["room", "day", "shift"]).cumcount() + 1
)



# ---------- 3) Capacity snapshots (final) ----------
# Rooms: free/used/utilization after the loop
rooms_free = df_capacity[["room", "day", "shift", "available", "free_min"]].copy()
rooms_free["cap_min"] = rooms_free["available"] * C_PER_SHIFT
rooms_free["used_min"] = (rooms_free["cap_min"] - rooms_free["free_min"]).clip(lower=0)
rooms_free["utilization"] = rooms_free.apply(
    lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0.0, axis=1
)
rooms_free = rooms_free.sort_values(["room", "day", "shift"]).reset_index(drop=True)

# Surgeons: remaining capacity by day/shift
surgeons_free = (
    df_surgeons[["surgeon_id", "day", "shift", "available"]]
    .drop_duplicates()
    .merge(df_surgeon_load[["surgeon_id", "day", "shift", "used_min"]],
           on=["surgeon_id", "day", "shift"], how="left")
    .fillna({"used_min": 0})
)
surgeons_free["cap_min"]  = surgeons_free["available"] * C_PER_SHIFT
surgeons_free["free_min"] = (surgeons_free["cap_min"] - surgeons_free["used_min"]).clip(lower=0)
surgeons_free["utilization"] = surgeons_free.apply(
    lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0.0, axis=1
)
surgeons_free = surgeons_free.sort_values(["surgeon_id", "day", "shift"]).reset_index(drop=True)

# ---------- 4) KPIs / summaries ----------
# Per-day KPIs (rooms)
kpi_day_rooms = rooms_free.groupby("day", as_index=False).agg(
    open_blocks=("available", "sum"),
    cap_min=("cap_min", "sum"),
    used_min=("used_min", "sum"),
    free_min=("free_min", "sum")
)
kpi_day_rooms["utilization"] = kpi_day_rooms.apply(
    lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0.0, axis=1
)

# Per-room KPIs (across all days/shifts)
kpi_room = rooms_free.groupby("room", as_index=False).agg(
    open_blocks=("available", "sum"),
    cap_min=("cap_min", "sum"),
    used_min=("used_min", "sum"),
    free_min=("free_min", "sum")
)
kpi_room["utilization"] = kpi_room.apply(
    lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0.0, axis=1
)

# Per-surgeon KPIs
kpi_surgeon = surgeons_free.groupby("surgeon_id", as_index=False).agg(
    open_blocks=("available", "sum"),
    cap_min=("cap_min", "sum"),
    used_min=("used_min", "sum"),
    free_min=("free_min", "sum")
)
kpi_surgeon["utilization"] = kpi_surgeon.apply(
    lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0.0, axis=1
)

# Global KPIs
total_cap = rooms_free["cap_min"].sum()
total_used = rooms_free["used_min"].sum()
total_free = rooms_free["free_min"].sum()
global_util = (total_used / total_cap) if total_cap > 0 else 0.0
n_assigned = len(assignments_enriched)
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
# remaining holds the patients not scheduled by the loop
assigned_ids = set(best_assignments["patient_id"])
unassigned_patients = df_patients[~df_patients["patient_id"].isin(assigned_ids)].copy()


# Optional: show their feasible blocks at the end (static view)
# (Rebuild Step-1 static feasibility just for these)
if len(unassigned_patients):
    blocks_open = df_rooms[df_rooms["available"] == 1][["room", "day", "shift"]].drop_duplicates()
    surg_open   = df_surgeons[df_surgeons["available"] == 1][["surgeon_id", "day", "shift"]].drop_duplicates()
    pmini_u = unassigned_patients[["patient_id", "surgeon_id", "duration"]].copy()
    ptime_u = pmini_u.merge(surg_open, on="surgeon_id", how="inner")
    pblocks_u = ptime_u.merge(blocks_open, on=["day", "shift"], how="inner")
    pblocks_u["fits_shift"] = (pblocks_u["duration"] + CLEANUP) <= C_PER_SHIFT
    pblocks_u = pblocks_u[pblocks_u["fits_shift"]]
    feas_u = (pblocks_u.groupby("patient_id", as_index=False)
                        .agg(feasible_blocks=("room", "count")))
    unassigned_patients = unassigned_patients.merge(feas_u, on="patient_id", how="left").fillna({"feasible_blocks":0})

# ---------- 6) Final block state (for audit) ----------
# One row per (room, day, shift) with remaining minutes (already in rooms_free)
# Join how many cases were assigned in each block
cases_per_block = (assignments_enriched.groupby(["room","day","shift"], as_index=False)
                                  .size().rename(columns={"size":"n_cases"}))
final_blocks = rooms_free.merge(cases_per_block, on=["room","day","shift"], how="left").fillna({"n_cases":0})


print("\n==================== FINAL SCHEDULE ====================\n")

if len(assignments_enriched) == 0:
    print("(No assignments found — nothing to display.)")
else:
    # ordenar por bloco e sequencia
    assignments_sorted = assignments_enriched.sort_values(
        ["room", "day", "shift", "seq_in_block", "patient_id"]
    )

    for (r, d, sh), group in assignments_sorted.groupby(["room", "day", "shift"], sort=True):
        # B_x_y_z em 0-based
        print(f"\nB_{int(r)-1}_{int(d)-1}_{int(sh)-1}:")

        for _, row in group.iterrows():
            pid = int(row["patient_id"])
            sid = int(row["surgeon_id"])
            dur = int(row["duration"])

            print(f"  (p={pid}, s={sid}, dur={dur})")

    print("\n========================================================\n")


# ---------- 7) Write everything to Excel ----------
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    # Inputs
    inputs_patients.to_excel(writer, sheet_name="Inputs_Patients", index=False)
    inputs_rooms.to_excel(writer, sheet_name="Inputs_Rooms", index=False)
    inputs_surgeons.to_excel(writer, sheet_name="Inputs_Surgeons", index=False)
    rooms_av_matrix.to_excel(writer, sheet_name="Rooms_Availability_Matrix", index=False)
    surgeons_av_matrix.to_excel(writer, sheet_name="Surgeons_Availability_Matrix", index=False)

    # Results
    assignments_enriched.to_excel(writer, sheet_name="Assignments", index=False)
    final_blocks.to_excel(writer, sheet_name="Blocks_FinalState", index=False)
    rooms_free.to_excel(writer, sheet_name="Rooms_Free", index=False)
    surgeons_free.to_excel(writer, sheet_name="Surgeons_Free", index=False)

    # KPIs
    kpi_global.to_excel(writer, sheet_name="KPI_Global", index=False)
    kpi_day_rooms.to_excel(writer, sheet_name="KPI_PerDay", index=False)
    kpi_room.to_excel(writer, sheet_name="KPI_PerRoom", index=False)
    kpi_surgeon.to_excel(writer, sheet_name="KPI_PerSurgeon", index=False)

    # Unassigned (if any)
    unassigned_patients.to_excel(writer, sheet_name="Unassigned", index=False)

print(f"\nExcel exported → {xlsx_path}")


