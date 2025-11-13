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
    surg_ok = df_surgeons[
        (df_surgeons["surgeon_id"] == sid) &
        (df_surgeons["available"] == 1)
    ][["day", "shift"]]

    # rooms open with enough capacity
    cap_ok = df_capacity[
        (df_capacity["available"] == 1) &
        (df_capacity["free_min"] >= need)
    ][["room", "day", "shift", "free_min"]]

    cand = surg_ok.merge(cap_ok, on=["day", "shift"], how="inner")

    # surgeon load within shift capacity
    surg_load = df_surgeon_load[
        df_surgeon_load["surgeon_id"] == sid
    ][["day", "shift", "used_min"]]

    cand = cand.merge(surg_load, on=["day", "shift"], how="left").fillna({"used_min": 0})
    cand = cand[(cand["used_min"] + need) <= C_PER_SHIFT]

    # continuity flag (already operating in same block)
    if len(df_assignments) > 0:
        cont = df_assignments[
            df_assignments["surgeon_id"] == sid
        ][["room", "day", "shift"]].copy()
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
    df["W_block"] = df["term_fit"] + df["term_early"] + df["term_cont"]
    return df.sort_values("W_block", ascending=False)


def commit_assignment(patient_row, best_row, iteration, w_patient=None, w_block=None):
    """Update capacity, surgeon-load, and record assignment with iteration order."""
    pid = int(patient_row["patient_id"])
    sid = int(patient_row["surgeon_id"])
    dur_need = int(patient_row["duration"]) + CLEANUP
    r, d, sh = int(best_row["room"]), int(best_row["day"]), int(best_row["shift"])

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
    return 3 if p == 3 else (15 if p == 2 else (90 if p == 1 else 270))

def deadline_term(priority, waited):
    lim = deadline_limit_from_priority(priority)
    if lim is None:
        return 0.0
    days_left = max(0, lim - waited)
    return 1.0 - (days_left / lim)


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
        (df_p_cap["free_min"] >= df_p_cap["need"]) &
        ((df_p_cap["used_min"] + df_p_cap["need"]) <= C_PER_SHIFT)
    ]
    
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
        print("\nNo more schedulable patients under Step-1 filters.")
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
        print(
            f"Iter {iteration:02d}: "
            f"Assign P{int(patient_row['patient_id'])} → "
            f"(Room={int(best_block['room'])}, Day={int(best_block['day'])}, Shift={int(best_block['shift'])}), "
            f"W_patient={patient_row['W_patient']:.4f}, W_block={best_block['W_block']:.3f}"
        )

        made_assignment = True
        break  # uma atribuição por iteração

    if not made_assignment:
        print("\nNo assignable patients under current Step-1 ranking (all fail Step-2). Stopping.")
        break

print("\nFinal assignments:")
print(df_assignments)


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

print("\nSurgeons — remaining free minutes per day/shift:")
print(df_surgeon_free.head(12))


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

print("\nRooms — remaining free minutes per day/shift:")
print(df_room_free.head(12))


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

print(f"\nExcel exported → {xlsx_path}")
