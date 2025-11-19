# -*- coding: utf-8 -*-
"""
Operating Room Scheduling — Step 1 + Step 2 (Iterative Dispatching Rule)
Author: Joana
"""

from pathlib import Path
import re
import ast
import pandas as pd
from datetime import datetime


"""" O ficheiro implementa o algoritmo de despacho em duas etapas (Step 1 + Step 2) para o problema de escalonamento de cirurgias em blocos operatórios:

Step 1 – escolher o próximo doente (dispatching rule ao nível do doente):avalia todos os doentes ainda por agendar, calcula um peso W_patient com base em prioridade, dias de espera, proximidade do deadline e escassez de blocos, ordena os doentes por esse peso.

Step 2 – escolher o melhor bloco para esse doente (dispatching rule ao nível do bloco/sala): para o doente escolhido, identifica todos os blocos (sala, dia, turno) em que ele é viável, aplica a regra do Cenário 1: se o cirurgião já estiver a operar num certo (dia, turno), tem de ficar na mesma sala durante esse turno,
calcula um peso W_block para cada bloco viável (encaixe, precocidade, continuidade),escolhe o bloco com maior W_block e fixa a cirurgia lá.

Repete esse ciclo até já não haver mais doentes que caibam em nenhum bloco.

No fim, calcula KPIs (utilização, capacidade usada, doentes agendados/por agendar) e exporta tudo para um ficheiro Excel.
"""


# ------------------------------
# PARAMETERS
# ------------------------------
DATA_FILE = "Instance_C1_30.DAT"

C_PER_SHIFT = 360   # minutes per shift (6h * 60)
CLEANUP = 17        # cleaning time

# weights
ALPHA1 = 0.25  # priority
ALPHA2 = 0.25  # waited days
ALPHA3 = 0.25  # deadline closeness
ALPHA4 = 0.25  # feasible blocks

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
df_patients = pd.DataFrame({                            #cria um dataframe com o id do paciente, com a respetiva duração estimada, prioridade, tempo que já esperou e o cirurugião alocado
    "patient_id": range(1, n_patients + 1),
    "duration": durations,
    "priority": priorities,
    "waiting": waitings,
    "surgeon_id": surgeons
})



# Rooms
rows = []
for r in range(n_rooms):                               #passar a matriz do block availability para um datframe -> room;day;shift;avalilabe    
    for d in range(n_days):
        for shift in (1, 2):  # 1=AM, 2=PM
            available = int(block_av[d][r][shift - 1])
            rows.append({"room": r + 1, "day": d + 1, "shift": shift, "available": available})
df_rooms = pd.DataFrame(rows)

# Surgeons
rows = []
for s in range(n_surgeons):                         #passar a matriz do surg_av para um dataframe -> surgeon_id;day;shift;available
    for d in range(n_days):
        for shift in (1, 2):
            availability = int(surg_av[s][d][shift - 1])
            rows.append({"surgeon_id": s + 1, "day": d + 1, "shift": shift, "available": availability})
df_surgeons = pd.DataFrame(rows)

# ------------------------------
# SUPPORT: deadline terms
# ------------------------------
def deadline_limit_from_priority(p):
    # tweakable limits
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
df_capacity["free_min"] = df_capacity["available"].apply(lambda a: C_PER_SHIFT if a == 1 else 0)

df_assignments = pd.DataFrame(
    columns=["patient_id", "room", "day", "shift", "used_min", "surgeon_id", "iteration", "W_patient", "W_block"]    #Tabela vazia onde vamos registar todas as consultas
)

df_surgeon_load = df_surgeons[["surgeon_id", "day", "shift"]].drop_duplicates().assign(used_min=0)                   #Adicionar used_min ao df_surgeons para percebermos quantos minutos cada cirurgão já trabalhou num determiando dia, num determinado turno.

# ------------------------------
# STEP 2 SUPPORT FUNCTIONS (Scenario 1 lock)
# ------------------------------
def feasible_blocks_step2(patient_row):
    """
    Return feasible (room, day, shift) for this patient given current capacity & surgeon load.
    Scenario 1: if the surgeon already has an assignment in (day, shift), they must stay in that same room.
    """
    sid = int(patient_row["surgeon_id"])
    need = int(patient_row["duration"]) + CLEANUP

    # surgeon availability (day, shift)
    surg_ok = df_surgeons[(df_surgeons["surgeon_id"] == sid) & (df_surgeons["available"] == 1)][["day", "shift"]].drop_duplicates()     #ver os dias e os turnos em que existem cirurgiões diposniveis para oa paciente ecnontrado no step1

    # rooms open with enough capacity
    cap_ok = df_capacity[(df_capacity["available"] == 1) & (df_capacity["free_min"] >= need)][["room", "day", "shift", "free_min"]].drop_duplicates()

    cand = surg_ok.merge(cap_ok, on=["day", "shift"], how="inner")                               #Ficar so com os dados de quando o cirurgiao esta disponivel e existe blocos disponiveis

    # surgeon load guard within shift capacity
    surg_load = df_surgeon_load[df_surgeon_load["surgeon_id"] == sid][["day", "shift", "used_min"]]
    cand = cand.merge(surg_load, on=["day", "shift"], how="left").fillna({"used_min": 0})                   
    cand = cand[(cand["used_min"] + need) <= C_PER_SHIFT]                                        #(minutos já usados pelo cirurgião no turno) + (minutos necessários para este doente) ≤ 360

    # Scenario 1: lock to room if surgeon already operating in that (day, shift) -> Isto garante que, se o cirurgião já está a operar numa sala naquele turno, todas as outras cirurgias nesse turno vão para a mesma sala.            
    if len(df_assignments) > 0:
        locks = (
            df_assignments[df_assignments["surgeon_id"] == sid]
            .loc[:, ["day", "shift", "room"]]
            .drop_duplicates()
            .rename(columns={"room": "room_locked"})
        )
        cand = cand.merge(locks, on=["day", "shift"], how="left")
        cand = cand[(cand["room_locked"].isna()) | (cand["room"] == cand["room_locked"])]
        cand = cand.drop(columns=["room_locked"])

        # continuity flag if already in same block
        cont = df_assignments[df_assignments["surgeon_id"] == sid][["room", "day", "shift"]].drop_duplicates() #Extrair cont com as combinações (room, day, shift) onde ele já foi marcado
        cont["continuity"] = 1
        cand = cand.merge(cont, on=["room", "day", "shift"], how="left")          # as linhas que batem certo com (room, day, shift) recebem 1; as outras ficam Na
        cand["continuity"] = cand["continuity"].fillna(0).astype(int)             
    else:
        cand["continuity"] = 0   # Se não há cirurgia continuity =0

    return cand

def score_block_for_patient(cand_df, patient_row, n_days):  #calcula o W_block de cada bloco candidato
    """Compute W_block for each candidate block."""
    need = int(patient_row["duration"]) + CLEANUP
    day_max = max(1, n_days - 1)
    df = cand_df.copy()
    df["free_after"] = (df["free_min"] - need).clip(lower=0)
    df["term_fit"]   = 1.0 - (df["free_after"] / C_PER_SHIFT)
    df["term_early"] = 1.0 - ((df["day"] - 1) / day_max)  # earlier days preferred
    df["term_cont"]  = df["continuity"].astype(float)     # stay in same block if possible
    df["W_block"] = df["term_fit"] + df["term_early"] + df["term_cont"]
    return df.sort_values("W_block", ascending=False)

def commit_assignment(patient_row, best_row, iteration, w_patient=None, w_block=None):   # atualiza o estado (capacidade + carga do cirurgião) e regista a cirurgia
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
    df_surgeon_load.loc[idx_s, "used_min"] += dur_need   #adiciona os minutos da respetiva cirurgia ao cirurgião

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

# ------------------------------
# ITERATIVE LOOP: Step 1 → Step 2 → commit → repeat
# ------------------------------
remaining = df_patients.copy()
iteration = 0

while True:                          # ciclo infinito que só é parado com break quando já não for possível marcar mais cirurgias
    iteration += 1   

    # ---- Step 1: dynamic feasible blocks per patient (with Scenario 1 lock) ----
    df_surg_open = df_surgeons[df_surgeons["available"] == 1][["surgeon_id", "day", "shift"]].drop_duplicates()
    df_cap_open  = df_capacity[df_capacity["available"] == 1][["room", "day", "shift", "free_min"]].drop_duplicates()
    df_sload     = df_surgeon_load[["surgeon_id", "day", "shift", "used_min"]].drop_duplicates()

    df_pmini = remaining[["patient_id", "surgeon_id", "duration", "priority", "waiting"]].copy()
    df_pmini["need"] = df_pmini["duration"] + CLEANUP

    # (surgeon availability)
    df_p_time = df_pmini.merge(df_surg_open, on="surgeon_id", how="inner")   # O paciente fica com todas as combinações de (day, shift) onde o seu cirurgião está disponível

    # (join current room capacity)
    df_p_cap = df_p_time.merge(df_cap_open, on=["day", "shift"], how="inner")  #Agora restringimos ainda mais para quando existem blocos e cirurgião dipsonivel

    # (current surgeon load per (day,shift))
    df_p_cap = df_p_cap.merge(df_sload, on=["surgeon_id", "day", "shift"], how="left").fillna({"used_min": 0})

    # keep only blocks that can host the case now
    df_p_blocks = df_p_cap[
        (df_p_cap["free_min"] >= df_p_cap["need"]) &
        ((df_p_cap["used_min"] + df_p_cap["need"]) <= C_PER_SHIFT)
    ]

    # Scenario 1 lock also in Step 1 feasibility counting
    if len(df_assignments) > 0:
        locks_all = (                                                            # este locks_all diz, para cada (cirurgião, dia, turno), se ele já está associado a uma sala
            df_assignments.loc[:, ["surgeon_id", "day", "shift", "room"]]
            .drop_duplicates()
            .rename(columns={"room": "room_locked"})
        )
        df_p_blocks = df_p_blocks.merge(locks_all, on=["surgeon_id", "day", "shift"], how="left")
        df_p_blocks = df_p_blocks[(df_p_blocks["room_locked"].isna()) | (df_p_blocks["room"] == df_p_blocks["room_locked"])]     #se o cirurgião ainda não tem sala naquele turno, entao qualquer sala serve. Caso contrário fica bloqueado a essa sala
        df_p_blocks = df_p_blocks.drop(columns=["room_locked"])

    # count feasible blocks per patient (after lock)
    df_feas_count = (
        df_p_blocks.groupby("patient_id", as_index=False)
                   .agg(feasible_blocks=("room", "count"))
    )

    step1 = df_pmini.merge(df_feas_count, on="patient_id", how="left").fillna({"feasible_blocks": 0})

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

    # NEW: iterate over the Step-1 ranking until we find someone who passes Step-2
    made_assignment = False
    step1_sorted = step1.sort_values("W_patient", ascending=False)

    for _, patient_row in step1_sorted.iterrows():
        # Step 2: compute feasible blocks (Scenario 1 lock enforced)
        cand_blocks = feasible_blocks_step2(patient_row)
        if cand_blocks.empty:
            # next candidate
            continue

        # score blocks and pick best
        scored = score_block_for_patient(cand_blocks, patient_row, n_days=n_days)
        best_block = scored.iloc[0]

        # commit
        commit_assignment(
            patient_row,
            best_block,
            iteration=iteration,
            w_patient=float(patient_row["W_patient"]),
            w_block=float(best_block["W_block"])
        )

        # remove scheduled patient
        remaining = remaining[remaining["patient_id"] != int(patient_row["patient_id"])]

        # progress log
        print(
            f"Iter {iteration:02d}: "
            f"Assign P{int(patient_row['patient_id'])} → "
            f"(Room={int(best_block['room'])}, Day={int(best_block['day'])}, Shift={int(best_block['shift'])}), "
            f"W_patient={patient_row['W_patient']:.4f}, W_block={best_block['W_block']:.3f}"
        )

        made_assignment = True
        break  # one assignment per iteration

    if not made_assignment:
        print("\nNo assignable patients under current Step-1 ranking (all fail Step-2). Stopping.")
        break

print("\nFinal assignments:")
print(df_assignments)

# --------------------------------------------
# PRETTY-PRINT: solution by block (Scenario 1 format)
# --------------------------------------------
assignments_enriched = df_assignments.merge(
    df_patients[["patient_id", "duration", "priority", "waiting"]],
    on="patient_id", how="left"
).sort_values("iteration")

ordered = assignments_enriched.sort_values(["day", "shift", "room", "iteration"])
schedule_by_block = {}
for (r, d, s), g in ordered.groupby(["room", "day", "shift"], sort=True):
    key = f"R{int(r)}_D{int(d)}_S{int(s)}"
    surgeon_id = int(g["surgeon_id"].iloc[0]) if len(g) else None  # scenario 1: single surgeon per block
    schedule_by_block[key] = {
        "surgeon": surgeon_id,
        "patients": [int(p) for p in g["patient_id"].tolist()]
    }

print("\n===== FINAL SCHEDULE BY BLOCK =====")
for key, val in sorted(schedule_by_block.items()):
    r = int(key.split("_")[0][1:])
    d = int(key.split("_")[1][1:])
    s = int(key.split("_")[2][1:])
    surgeon_id = val["surgeon"]
    patients_str = ", ".join(f"P{p}" for p in val["patients"])
    print(f"Room {r}, Day {d}, Shift {s} — Surgeon {surgeon_id} → {patients_str}")
print("===================================")

# --------------------------------------------
# SURGEONS: remaining free minutes per day/shift
# --------------------------------------------
df_surgeon_free = (
    df_surgeons[["surgeon_id", "day", "shift", "available"]]
    .drop_duplicates()
    .merge(df_surgeon_load[["surgeon_id", "day", "shift", "used_min"]],
           on=["surgeon_id", "day", "shift"], how="left")
    .fillna({"used_min": 0})
)
df_surgeon_free["cap_min"]  = df_surgeon_free["available"] * C_PER_SHIFT
df_surgeon_free["free_min"] = (df_surgeon_free["cap_min"] - df_surgeon_free["used_min"]).clip(lower=0)
df_surgeon_free["utilization"] = df_surgeon_free.apply(
    lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0.0, axis=1
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
    lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0.0, axis=1
)
df_room_free = df_room_free.sort_values(["room", "day", "shift"]).reset_index(drop=True)

print("\nRooms — remaining free minutes per day/shift:")
print(df_room_free.head(12))

# ============================================================
# EXPORT PACK — build all relevant tables and write to Excel
# ============================================================
ts = datetime.now().strftime("%Y%m%d_%H%M")
xlsx_path = f"or_schedule_export_{ts}.xlsx"

# Inputs
inputs_patients = df_patients.sort_values("patient_id").copy()
inputs_rooms = df_rooms.sort_values(["room", "day", "shift"]).copy()
inputs_surgeons = df_surgeons.sort_values(["surgeon_id", "day", "shift"]).copy()

rooms_av_matrix = inputs_rooms.pivot_table(
    index=["room", "day"], columns="shift", values="available", aggfunc="first"
).rename(columns={1: "AM", 2: "PM"}).reset_index()

surgeons_av_matrix = inputs_surgeons.pivot_table(
    index=["surgeon_id", "day"], columns="shift", values="available", aggfunc="first"
).rename(columns={1: "AM", 2: "PM"}).reset_index()

# Results already have assignments_enriched ordered by iteration
# Add a simple sequence number per (room,day,shift)
assignments_enriched = assignments_enriched.copy()
assignments_enriched["seq_in_block"] = (
    assignments_enriched.groupby(["room", "day", "shift"]).cumcount() + 1
)

# Capacity snapshots (final)
rooms_free = df_room_free.copy()
surgeons_free = df_surgeon_free.copy()

# KPIs / summaries
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
n_unassigned = len(df_patients) - n_assigned

kpi_global = pd.DataFrame([{
    "total_capacity_min": total_cap,
    "total_used_min": total_used,
    "total_free_min": total_free,
    "global_utilization": global_util,
    "assigned_patients": n_assigned,
    "unassigned_patients": n_unassigned
}])

# Unassigned patients
unassigned_patients = remaining.sort_values("patient_id").copy() if len(remaining) else pd.DataFrame(
    columns=df_patients.columns
)

# Final block state (for audit)
cases_per_block = (
    assignments_enriched.groupby(["room", "day", "shift"], as_index=False)
    .size()
    .rename(columns={"size": "n_cases"})
)
final_blocks = rooms_free.merge(cases_per_block, on=["room", "day", "shift"], how="left").fillna({"n_cases": 0})

# Write Excel
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
