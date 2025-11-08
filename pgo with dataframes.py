# -*- coding: utf-8 -*-
"""
Operating Room Scheduling — Step 1 + Step 2 (Iterative Dispatching Rule)
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

ALPHA1 = 0.45  # priority
ALPHA2 = 0.15  # waited days
ALPHA3 = 0.35  # deadline closeness
ALPHA4 = 0.05  # feasible blocks


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
    cand = cand[(cand["used_min"] + need) <= C_PER_SHIFT]

    # continuity flag (already operating in same block)
    if len(df_assignments) > 0:
        cont = df_assignments[df_assignments["surgeon_id"] == sid][["room", "day", "shift"]].copy()
        cont["continuity"] = 1
        cand = cand.merge(cont, on=["room", "day", "shift"], how="left")
    else:
        cand["continuity"] = 0

    cand["continuity"] = cand["continuity"].fillna(0).astype(int)
    print(f"surgeon: {sid}, need: {need}")
    print(cand)
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
    df["W_block"] = df["term_fit"] + df["term_early"] + df["term_cont"]
    return df.sort_values("W_block", ascending=False)


def commit_assignment(patient_row, best_row):
    """Update capacity, surgeon-load, and assignments after scheduling."""
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

    # record assignment
    df_assignments.loc[len(df_assignments)] = {
        "patient_id": pid, "room": r, "day": d, "shift": sh,
        "used_min": dur_need, "surgeon_id": sid
    }


def deadline_limit_from_priority(p):
    return 15 if p == 2 else (270 if p == 1 else None)

def deadline_term(priority, waited):
    lim = deadline_limit_from_priority(priority)
    if lim is None: return 0.0
    days_left = max(0, lim - waited)
    return 1.0 - (days_left / lim)


# ------------------------------
# INITIAL PLANNING STATE
# ------------------------------
df_capacity = df_rooms.copy()
df_capacity["free_min"] = df_capacity["available"].apply(lambda a: C_PER_SHIFT if a == 1 else 0)

df_assignments = pd.DataFrame(columns=["patient_id", "room", "day", "shift", "used_min", "surgeon_id"])

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
                           ((df_p_cap["used_min"] + df_p_cap["need"]) <= C_PER_SHIFT)]
    
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

    # commit assignment
    commit_assignment(patient_row, best_block)

    # remove scheduled patient
    remaining = remaining[remaining["patient_id"] != patient_row["patient_id"]]

    # progress log
    print(f"Iter {iteration:02d}: "
          f"Assign P{int(patient_row['patient_id'])} → "
          f"(Room={int(best_block['room'])}, Day={int(best_block['day'])}, Shift={int(best_block['shift'])}), "
          f"W_patient={patient_row['W_patient']:.4f}, W_block={best_block['W_block']:.3f}")

print("\nFinal assignments:")
print(df_assignments)













































