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
DATA_FILE = "instance_c1_30.dat"

C_PER_SHIFT = 360   # minutes per shift (6h * 60)
CLEANUP = 17        # cleaning time 

ALPHA1 = 0.25  # priority
ALPHA2 = 0.25  # waited days
ALPHA3 = 0.25  # deadline closeness
ALPHA4 = 0.25  # feasible blocks

ALPHA5 = 0.25  # priority
ALPHA6 = 0.25  # waited days
ALPHA7 = 0.25 

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

def evaluate_schedule(assignments, patients, rooms_free, weights=(0.4, 0.3, 0.2, 0.1)):
    w1, w2, w3, w4 = weights
    total_patients = len(patients)
    ratio_scheduled = (len(assignments) / total_patients) if total_patients else 0.0

    util_rooms = float(rooms_free.loc[rooms_free["cap_min"] > 0, "utilization"].mean()) if len(rooms_free) else 0.0

    if len(assignments):
        merged = assignments.merge(
            patients[["patient_id","priority","waiting"]],
            on="patient_id", how="left"
        )
        prio_rate = float((merged["priority"] > 0).mean())
        wmax = float(patients["waiting"].max())
        norm_wait_term = 1.0 - float(merged["waiting"].mean() / wmax) if wmax > 0 else 1.0
    else:
        prio_rate = 0.0
        norm_wait_term = 0.0

    score = (w1*ratio_scheduled + w2*util_rooms + w3*prio_rate + w4*norm_wait_term)
    return {"score":float(score), "ratio_scheduled":float(ratio_scheduled),
            "util_rooms":float(util_rooms), "prio_rate":float(prio_rate),
            "norm_wait_term":float(norm_wait_term)}




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


def swap_assigned_random(assignments, df_rooms, df_surgeons, C_PER_SHIFT):
    """
    Swap between two already assigned patients, if feasible.
    """
    if len(assignments) < 2:
        return assignments, False, None

    new_assign = assignments.copy()
    # escolher dois pacientes aleatórios
    sampled = new_assign.sample(2)
    p1 = sampled.iloc[0]
    p2 = sampled.iloc[1]

    # blocos atuais
    r1, d1, sh1 = int(p1['room']), int(p1['day']), int(p1['shift'])
    r2, d2, sh2 = int(p2['room']), int(p2['day']), int(p2['shift'])

    need1 = int(p1['used_min'])
    need2 = int(p2['used_min'])

    #if p1['room'] != r2:
        #need1 += ROOM_CHANGE_TIME
    #if p2['room'] != r1:
        #need2 += ROOM_CHANGE_TIME

    # calcular minutos livres nos blocos se retirarmos cada paciente
    block1_free = C_PER_SHIFT - (new_assign[(new_assign['room']==r1)&(new_assign['day']==d1)&(new_assign['shift']==sh1)]['used_min'].sum() - need1)
    block2_free = C_PER_SHIFT - (new_assign[(new_assign['room']==r2)&(new_assign['day']==d2)&(new_assign['shift']==sh2)]['used_min'].sum() - need2)

    # checar viabilidade
    if need2 <= block1_free and need1 <= block2_free:
        # aplicar swap
        new_assign.loc[new_assign['patient_id']==p1['patient_id'], ['room','day','shift']] = [r2,d2,sh2]
        new_assign.loc[new_assign['patient_id']==p2['patient_id'], ['room','day','shift']] = [r1,d1,sh1]
        return new_assign, True, (p1['patient_id'], p2['patient_id'], r1,d1,sh1, r2,d2,sh2)
    else:
        return assignments, False, None



def local_search_iterated(assign_init, df_rooms, df_surgeons, patients, C_PER_SHIFT, max_no_improv=200):
    current = assign_init.copy()

    # avaliar inicial
    rooms_base = df_rooms[["room","day","shift","available"]].copy()
    rooms_base["cap_min"] = rooms_base["available"] * C_PER_SHIFT
    if len(current):
        used_by_block = (current.groupby(["room","day","shift"], as_index=False)
                         .agg(used_min=("used_min","sum")))
    else:
        used_by_block = rooms_base[["room","day","shift"]].copy().assign(used_min=0)
    rooms_join = rooms_base.merge(used_by_block, on=["room","day","shift"], how="left").fillna({"used_min":0})
    rooms_join["used_min"] = rooms_join["used_min"].clip(lower=0)
    rooms_join["utilization"] = rooms_join.apply(lambda r: (r["used_min"]/r["cap_min"]) if r["cap_min"]>0 else 0.0, axis=1)

    feas = feasibility_metrics(current, df_rooms, df_surgeons, patients, C_PER_SHIFT)
    best_feas = feas["feasibility_score"]
    if best_feas == 0:
        best_score = evaluate_schedule(current, patients, rooms_join)["score"]
    else:
        best_score = -best_feas

    print(f"\nLS START — score inicial={best_score:.4f}, infeas={best_feas}")

    no_improv = 0
    ls_iter = 0

    while no_improv < max_no_improv:
        ls_iter += 1

        candidate, moved, swap_info = swap_assigned_random(current, df_rooms, df_surgeons, C_PER_SHIFT)
        if not moved:
            no_improv += 1
            continue

        feas_c = feasibility_metrics(candidate, df_rooms, df_surgeons, patients, C_PER_SHIFT)

        rooms_base = df_rooms[["room","day","shift","available"]].copy()
        rooms_base["cap_min"] = rooms_base["available"] * C_PER_SHIFT
        if len(candidate):
            used_by_block = (candidate.groupby(["room","day","shift"], as_index=False)
                             .agg(used_min=("used_min","sum")))
        else:
            used_by_block = rooms_base[["room","day","shift"]].copy().assign(used_min=0)
        rooms_join_c = rooms_base.merge(used_by_block, on=["room","day","shift"], how="left").fillna({"used_min":0})
        rooms_join_c["used_min"] = rooms_join_c["used_min"].clip(lower=0)
        rooms_join_c["utilization"] = rooms_join_c.apply(lambda r: (r["used_min"]/r["cap_min"]) if r["cap_min"]>0 else 0.0, axis=1)

        if feas_c["feasibility_score"] == 0:
            cand_score = evaluate_schedule(candidate, patients, rooms_join_c)["score"]
        else:
            cand_score = -feas_c["feasibility_score"]


        if cand_score > best_score:
            print(f"[LS] MELHORIA: {best_score:.4f} → {cand_score:.4f}")
            # NEW
            p1_id, p2_id, r1,d1,sh1, r2,d2,sh2 = swap_info
            print(f"   swap: P{p1_id} (B_{r1}_{d1}_{sh1})  ↔  P{p2_id} (B_{r2}_{d2}_{sh2})")

            print("   ✓ melhoria — movimento aceite")
            current = candidate
            best_score = cand_score
            best_feas = feas_c["feasibility_score"]
            no_improv = 0
        else:
            no_improv += 1

    print(f"\nLS END — melhor score={best_score:.4f}")
    return current, best_score, best_feas



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

    # ======== CORREÇÃO: varrer o ranking inteiro ========
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

#print("\nFinal assignments:")
#print(df_assignments)


#print("\n==================== SOLUÇÃO INICIAL (Step1+Step2) ====================\n")

#if df_assignments.empty:
#    print("(No assignments yet — run Step1+Step2 first.)")
#else:
#    initial_sorted = df_assignments.sort_values(["day", "shift", "room"])
#    for _, row in initial_sorted.iterrows():
#        print(
#            f"P{int(row['patient_id'])} → Room {int(row['room'])}, "
#            f"Day {int(row['day'])}, Shift {int(row['shift'])}, "
#            f"Used_min {int(row['used_min'])}, Surgeon {int(row['surgeon_id'])}"
#        )


# ========================
# LOCAL SEARCH ITERATED NEW
# ========================
#print("\n>> Running local search iterated (swap_with_unassigned_random only)...")
#best_assignments, best_score, best_feas = local_search_iterated(
#    df_assignments, df_rooms, df_surgeons, df_patients, C_PER_SHIFT, max_no_improv=200
#)

#print("\nLocal search finished:",
#      f"best_score={best_score:.4f}, best_feas={best_feas}")

# usar a solução melhorada (ou manter original se não melhorou)
#if len(best_assignments):
#    df_assignments = best_assignments.copy()
#else:
#    print("No better solution found; keeping constructive solution.")


#NEW
# rebuild df_capacity and df_surgeon_load from the final df_assignments
# reset df_capacity free_min
df_capacity = df_rooms.copy()
df_capacity["free_min"] = df_capacity["available"].apply(lambda a: C_PER_SHIFT if a == 1 else 0)

# subtract used minutes per block
if len(df_assignments):
    used_by_block = df_assignments.groupby(["room","day","shift"], as_index=False).agg(used_min=("used_min","sum"))
    for _, r in used_by_block.iterrows():
        idx = (df_capacity["room"]==int(r["room"])) & (df_capacity["day"]==int(r["day"])) & (df_capacity["shift"]==int(r["shift"]))
        df_capacity.loc[idx, "free_min"] -= int(r["used_min"])

# rebuild surgeon load
df_surgeon_load = df_surgeons[["surgeon_id","day","shift"]].drop_duplicates().assign(used_min=0)
if len(df_assignments):
    sload = df_assignments.groupby(["surgeon_id","day","shift"], as_index=False).agg(used_min=("used_min","sum"))
    for _, r in sload.iterrows():
        idx_s = (df_surgeon_load["surgeon_id"]==int(r["surgeon_id"])) & (df_surgeon_load["day"]==int(r["day"])) & (df_surgeon_load["shift"]==int(r["shift"]))
        df_surgeon_load.loc[idx_s, "used_min"] = int(r["used_min"])


# --------------------------------------------
# SURGEONS: remaining free minutes per day/shift
# --------------------------------------------
df_surgeon_free = (
    df_surgeons[["surgeon_id", "day", "shift", "available"]]
    .drop_duplicates()
    .merge(
        df_surgeon_load[["surgeon_id", "day", "shift", "used_min"]],
        on=["surgeon_id", "day", "shift"],
        how="left"
    )
    .fillna({"used_min": 0})
)

df_surgeon_free["cap_min"]  = df_surgeon_free["available"] * C_PER_SHIFT
df_surgeon_free["free_min"] = (df_surgeon_free["cap_min"] - df_surgeon_free["used_min"]).clip(lower=0)

df_surgeon_free["utilization"] = df_surgeon_free.apply(
    lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0.0,
    axis=1
)

df_surgeon_free = df_surgeon_free.sort_values(["surgeon_id", "day", "shift"]).reset_index(drop=True)

#print("\nSurgeons — remaining free minutes per day/shift:")
#print(df_surgeon_free.head(12))


# --------------------------------------------
# ROOMS: remaining free minutes per day/shift
# --------------------------------------------
df_room_free = df_capacity[["room", "day", "shift", "available", "free_min"]].copy()

df_room_free["cap_min"]   = df_room_free["available"] * C_PER_SHIFT
df_room_free["used_min"]  = (df_room_free["cap_min"] - df_room_free["free_min"]).clip(lower=0)
df_room_free["utilization"] = df_room_free.apply(
    lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0.0,
    axis=1
)

df_room_free = df_room_free.sort_values(["room", "day", "shift"]).reset_index(drop=True)

#print("\nRooms — remaining free minutes per day/shift:")
#print(df_room_free.head(12))


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

# ---------- 2) Assignments enriched ----------
assignments_enriched = df_assignments.merge(
    df_patients[["patient_id", "duration", "priority", "waiting"]],
    on="patient_id",
    how="left"
).sort_values("iteration")

assignments_enriched["seq_in_block"] = (
    assignments_enriched.groupby(["room", "day", "shift"]).cumcount() + 1
)

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

n_assigned = len(assignments_enriched)
assigned_ids = set(assignments_enriched["patient_id"].unique())
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
    assignments_enriched.groupby(["room", "day", "shift"], as_index=False)
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

#print(f"\nExcel exported → {xlsx_path}")


# ---------- 8) TEXT-BASED SCHEDULE (formato tipo imagem) ----------

print("\n==================== FINAL SCHEDULE ====================\n")

if len(assignments_enriched) == 0:
    print("(No assignments found — nothing to display.)")
else:
    # AGORA vamos incluir a limpeza na timeline
    INCLUDE_CLEANUP_IN_TIMELINE = True

    # Ordenar blocos e casos
    assignments_sorted = assignments_enriched.sort_values(
        ["day", "shift", "room", "seq_in_block"]
    )

    for (r, d, sh), group in assignments_sorted.groupby(
        ["room", "day", "shift"], sort=True
    ):
        # Header do bloco (0-based, como na imagem)
        print(f"\nB_{int(r)-1}_{int(d)-1}_{int(sh)-1}:")

        t = 0  # timeline accumulator (minutos desde início do shift)

        for _, row in group.iterrows():
            pid = int(row["patient_id"])
            sid = int(row["surgeon_id"])
            dur = int(row["duration"])

            start = t
            end = t + dur  # fim da cirurgia (não inclui limpeza)

            print(f"  (p={pid}, s={sid}, dur={dur}, start={start}, end={end})")

            # avanço do relógio: soma duração + CLEANUP
            t = end + (CLEANUP if INCLUDE_CLEANUP_IN_TIMELINE else 0)

    print("\n========================================================\n")


# --------------------------------------------
# RUN FEASIBILITY + EVALUATION ON FINAL SOLUTION 
# --------------------------------------------
feas = feasibility_metrics(
    assignments=df_assignments,
    df_rooms=df_rooms,
    df_surgeons=df_surgeons,
    patients=df_patients,
    C_PER_SHIFT=C_PER_SHIFT
)

print("\nFeasibility:", feas["feasibility_score"],
      "| unassigned:", feas["n_unassigned"],
      "| block_closed:", feas["block_unavailable_viol"],
      "| surg_unavail:", feas["surg_unavailable_viol"],
      "| excess_block_min:", feas["excess_block_min"],
      "| excess_surgeon_min:", feas["excess_surgeon_min"])

# EVALUATE QUALITY
ev = {}
if feas["feasibility_score"] == 0:
    ev = evaluate_schedule(df_assignments, df_patients, rooms_free)
    print("Evaluation score:", f"{ev['score']:.4f}",
          "| scheduled:", f"{ev['ratio_scheduled']:.3f}",
          "| util:", f"{ev['util_rooms']:.3f}",
          "| prio:", f"{ev['prio_rate']:.3f}",
          "| wait_term:", f"{ev['norm_wait_term']:.3f}")

    print("Feasibility penalty (soft):",
          "excess_block_min =", feas["excess_block_min"],
          "| excess_surgeon_min =", feas["excess_surgeon_min"])

# EXCEL – folha de avaliação
_eval_row = {
    "n_unassigned": feas["n_unassigned"],
    "block_unavailable_viol": feas["block_unavailable_viol"],
    "surg_unavailable_viol": feas["surg_unavailable_viol"],
    "excess_block_min": feas["excess_block_min"],
    "excess_surgeon_min": feas["excess_surgeon_min"],
    "feasibility_score": feas["feasibility_score"],
}
_eval_row.update({k: ev.get(k, None) for k in ["score","ratio_scheduled","util_rooms","prio_rate","norm_wait_term"]})
eval_df = pd.DataFrame([_eval_row])

from openpyxl import load_workbook  # só para garantir que o engine existe

with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    eval_df.to_excel(writer, sheet_name="Feasibility_Eval", index=False)
