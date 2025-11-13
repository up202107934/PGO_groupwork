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
# FEASIBILITY & EVALUATION FUNCTIONS (para local search)
# --------------------------------------------

def feasibility_metrics(assignments, df_rooms, df_surgeons, patients, C_PER_SHIFT):
    """
    Mede quão 'inviável' é uma solução:
    - excesso de minutos por bloco
    - excesso de minutos por cirurgião
    - cirurgias em blocos fechados
    - cirurgião em bloco onde não está disponível
    """
    # capacidade base por bloco
    rooms_base = df_rooms[["room", "day", "shift", "available"]].copy()
    rooms_base["cap_min"] = rooms_base["available"] * C_PER_SHIFT

    # uso por bloco (a partir de assignments)
    if len(assignments):
        used_by_block = (assignments.groupby(["room", "day", "shift"], as_index=False)
                         .agg(used_min=("used_min", "sum")))
    else:
        used_by_block = rooms_base[["room","day","shift"]].copy()
        used_by_block["used_min"] = 0

    rooms_join = rooms_base.merge(
        used_by_block,
        on=["room","day","shift"],
        how="left"
    ).fillna({"used_min": 0})

    rooms_join["excess_min"] = (rooms_join["used_min"] - rooms_join["cap_min"]).clip(lower=0)
    excess_block_min = int(rooms_join["excess_min"].sum())

    # blocos fechados com cirurgias
    bad_block_assigns = assignments.merge(
        rooms_base,
        on=["room","day","shift"],
        how="left"
    )
    block_unavailable_viol = int((bad_block_assigns["available"].fillna(0) == 0).sum())

    # disponibilidade de cirurgião por (day, shift)
    surg_base = df_surgeons[["surgeon_id", "day", "shift", "available"]].drop_duplicates()
    ass_with_surg = assignments.merge(
        surg_base,
        on=["surgeon_id","day","shift"],
        how="left",
        suffixes=("","_s")
    )
    surg_unavailable_viol = int((ass_with_surg["available"].fillna(0) == 0).sum())

    # excesso por cirurgião em cada (day,shift)
    if len(assignments):
        sload = (assignments.groupby(["surgeon_id","day","shift"], as_index=False)
                 .agg(used_min=("used_min","sum")))
        sload["excess_min"] = (sload["used_min"] - C_PER_SHIFT).clip(lower=0)
        excess_surgeon_min = int(sload["excess_min"].sum())
    else:
        excess_surgeon_min = 0

    feasibility_score = (
        block_unavailable_viol +
        surg_unavailable_viol +
        excess_block_min +
        excess_surgeon_min
    )

    return {
        "feasibility_score": feasibility_score,
        "excess_block_min": excess_block_min,
        "excess_surgeon_min": excess_surgeon_min,
        "rooms_join": rooms_join
    }


def evaluate_schedule(assignments, patients, rooms_join,
                      weights=(0.4, 0.3, 0.2, 0.1)):
    """
    Score de qualidade da solução:
    - % de doentes agendados
    - utilização média das salas
    - cobertura de prioridades
    - termo de tempo de espera médio
    """
    w1, w2, w3, w4 = weights
    total_patients = len(patients)
    ratio_scheduled = len(assignments) / total_patients if total_patients else 0.0

    # utilização média das salas
    if len(rooms_join):
        total_cap = rooms_join["cap_min"].sum()
        total_used = rooms_join["used_min"].sum()
        util_rooms = float(total_used / total_cap) if total_cap > 0 else 0.0
    else:
        util_rooms = 0.0

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

    score = (w1*ratio_scheduled +
             w2*util_rooms +
             w3*prio_rate +
             w4*norm_wait_term)

    return float(score)

import random  # garante que tens o import no topo do ficheiro


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


def move_patient(assignments, patients, df_rooms, df_surgeons, C_PER_SHIFT):
    """
    Tenta gerar um vizinho movendo 1 doente para outro bloco.
    Devolve (new_assignments, moved) onde moved=True se conseguiu.
    """
    if assignments.empty:
        return assignments, False

    # escolher paciente aleatório da solução atual
    pid = int(random.choice(assignments["patient_id"].tolist()))
    patient_info = patients[patients["patient_id"] == pid].iloc[0]

    # blocos candidatos na solução atual
    cand_blocks = candidate_blocks_for_patient_in_solution(
        assignments,
        patient_info,
        df_rooms,
        df_surgeons,
        C_PER_SHIFT
    )

    # remover o bloco onde já está
    current_row = assignments[assignments["patient_id"] == pid].iloc[0]
    cand_blocks = cand_blocks[~(
        (cand_blocks["room"]  == current_row["room"]) &
        (cand_blocks["day"]   == current_row["day"]) &
        (cand_blocks["shift"] == current_row["shift"])
    )]

    if cand_blocks.empty:
        return assignments, False

    # escolher 1 bloco aleatório
    chosen = cand_blocks.sample(1).iloc[0]

    # construir novo df_assignments com o paciente movido
    new_assign = assignments.copy()

    idx_old = new_assign.index[new_assign["patient_id"] == pid]
    new_assign.loc[idx_old, ["room","day","shift"]] = (
        int(chosen["room"]), int(chosen["day"]), int(chosen["shift"])
    )

    # opcional: reordenar por bloco
    new_assign = new_assign.sort_values(["day","shift","room","iteration"]).reset_index(drop=True)

    return new_assign, True


def swap_patients(assignments, patients, df_rooms, df_surgeons, C_PER_SHIFT):
    """
    N₂: troca dois doentes de bloco (room, day, shift),
    mas APENAS entre doentes do MESMO cirurgião.
    Agora com debug prints para ver o que está a acontecer.
    """
    print("\n===== DEBUG: swap_patients =====")

    if len(assignments) < 2:
        print("➡️  Menos de 2 assignments → impossível trocar.")
        return assignments, False

    new_assign = assignments.copy()

    # 1) cirurgiões que têm pelo menos 2 doentes agendados
    surg_counts = new_assign.groupby("surgeon_id")["patient_id"].nunique()
    eligible_surgeons = surg_counts[surg_counts >= 2].index.tolist()
    print(f"Cirurgiões elegíveis para swap (>=2 pacientes): {eligible_surgeons}")

    if not eligible_surgeons:
        print("➡️  Nenhum cirurgião tem 2 doentes → swap impossível.")
        return assignments, False

    # escolhe 1 cirurgião aleatório
    sid = random.choice(eligible_surgeons)
    print(f"🎯 Cirurgião escolhido: {sid}")

    sub = new_assign[new_assign["surgeon_id"] == sid]
    patient_ids = sub["patient_id"].unique()
    print(f"Pacientes do cirurgião {sid}: {patient_ids}")

    if len(patient_ids) < 2:
        print("➡️  Menos de 2 doentes deste cirurgião → impossível trocar.")
        return assignments, False

    # 2) tentar escolher 2 doentes em blocos diferentes
    max_tries = 20
    for attempt in range(1, max_tries + 1):
        print(f"\n--- Tentativa {attempt}/{max_tries} ---")

        pid1, pid2 = random.sample(list(patient_ids), 2)
        print(f"Tentativa de swap entre P{pid1} e P{pid2}")

        row1 = sub.loc[sub["patient_id"] == pid1, ["room", "day", "shift"]].iloc[0].copy()
        row2 = sub.loc[sub["patient_id"] == pid2, ["room", "day", "shift"]].iloc[0].copy()

        print(f"P{pid1} está em (R{row1['room']},D{row1['day']},S{row1['shift']})")
        print(f"P{pid2} está em (R{row2['room']},D{row2['day']},S{row2['shift']})")

        # se estiverem no mesmo bloco → tentar outro par
        if (row1["room"] == row2["room"]) and \
           (row1["day"] == row2["day"]) and \
           (row1["shift"] == row2["shift"]):
            print("⚠️  Estão no MESMO bloco → tentar outra combinação.")
            continue

        print("✅ Blocos diferentes → pode fazer swap.")

        # 3) aplicar swap
        new_assign.loc[new_assign["patient_id"] == pid1, ["room", "day", "shift"]] = row2.values
        new_assign.loc[new_assign["patient_id"] == pid2, ["room", "day", "shift"]] = row1.values

        print(f"✔️ Swap efetuado:")
        print(f"   P{pid1} → (R{row2['room']},D{row2['day']},S{row2['shift']})")
        print(f"   P{pid2} → (R{row1['room']},D{row1['day']},S{row1['shift']})")

        # reordenar
        new_assign = new_assign.sort_values(["day", "shift", "room", "iteration"]).reset_index(drop=True)

        print("➡️  Swap concluído com sucesso.")
        return new_assign, True

    print("❌ Não foi possível encontrar dois pacientes em blocos diferentes após várias tentativas.")
    return assignments, False


def swap_with_unassigned(assignments, patients, df_rooms, df_surgeons, C_PER_SHIFT):
    """
    N3: tenta inserir o paciente NÃO agendado mais urgente (priority, waiting)
    e retirar o paciente MENOS urgente de um bloco onde isto seja possível.
    Este vizinho altera o conjunto de pacientes agendados → mexe no score.
    """
    new_assign = assignments.copy()

    # pacientes agendados
    scheduled_ids = set(new_assign["patient_id"])

    # pacientes não agendados
    unassigned = patients[~patients["patient_id"].isin(scheduled_ids)]
    if unassigned.empty:
        return assignments, False

    # 1) escolher o não-agendado MAIS urgente
    u_row = unassigned.sort_values(
        ["priority", "waiting"], ascending=[False, False]
    ).iloc[0]
    u_id = int(u_row["patient_id"])
    u_sid = int(u_row["surgeon_id"])
    need = int(u_row["duration"]) + CLEANUP

    # 2) blocos onde o cirurgião está disponível
    surg_ok = df_surgeons[
        (df_surgeons["surgeon_id"] == u_sid) &
        (df_surgeons["available"] == 1)
    ][["day","shift"]]

    # capacidade base por bloco
    rooms_base = df_rooms[["room","day","shift","available"]].copy()
    rooms_base["cap_min"] = rooms_base["available"] * C_PER_SHIFT

    # uso atual
    used_by_block = new_assign.groupby(
        ["room","day","shift"], as_index=False
    ).agg(used_min=("used_min","sum"))
    
    rooms_join = rooms_base.merge(
        used_by_block, on=["room","day","shift"], how="left"
    ).fillna({"used_min":0})

    rooms_join["free_min"] = (
        rooms_join["cap_min"] - rooms_join["used_min"]
    ).clip(lower=0)

    # 3) blocos candidatos:
    #    onde u PODERIA CABER SE tirarmos alguém
    cand_blocks = surg_ok.merge(
        rooms_join[rooms_join["available"]==1][["room","day","shift","used_min","cap_min","free_min"]],
        on=["day","shift"], how="inner"
    )

    if cand_blocks.empty:
        return assignments, False

    # escolher um bloco ao acaso
    b = cand_blocks.sample(1).iloc[0]
    r, d, sh = int(b["room"]), int(b["day"]), int(b["shift"])

    # pacientes nesse bloco
    in_block = new_assign[
        (new_assign["room"] == r) &
        (new_assign["day"] == d) &
        (new_assign["shift"] == sh)
    ].merge(
        patients[["patient_id","priority","waiting","duration"]],
        on="patient_id", how="left"
    )

    if in_block.empty:
        return assignments, False

    # 4) escolher o paciente MENOS urgente para sair
    out_row = in_block.sort_values(
        ["priority","waiting"], ascending=[True, True]
    ).iloc[0]

    out_id = int(out_row["patient_id"])
    out_dur = int(out_row["duration"]) + CLEANUP


    # 5) verificar capacidade se fizermos a troca
    used_now = int(b["used_min"])
    cap = int(b["cap_min"])

    if used_now - out_dur + need > cap:
        return assignments, False
    
    # ========== PRINTS APENAS A PARTIR DAQUI (swap vai acontecer) ==========
    print(f"\n🔄 SWAP WITH UNASSIGNED:")
    print(f" → Inserir P{u_id} (prio={u_row['priority']}, wait={u_row['waiting']})")
    print(f" → Retirar P{out_id} (prio={out_row['priority']}, wait={out_row['waiting']})")
    print(f" → No bloco (Room={r}, Day={d}, Shift={sh})")
    
    # 6) aplicar a troca
    new_assign = new_assign[new_assign["patient_id"] != out_id].copy()

    new_assign = pd.concat([
        new_assign,
        pd.DataFrame([{
            "patient_id": u_id,
            "room": r,
            "day": d,
            "shift": sh,
            "used_min": need,
            "surgeon_id": u_sid,
            "iteration": new_assign["iteration"].max() + 1 if len(new_assign) else 1,
            "W_patient": None,
            "W_block": None
        }])
    ], ignore_index=True)

    new_assign = new_assign.sort_values(
        ["day","shift","room","iteration"]).reset_index(drop=True)

    return new_assign, True

def swap_with_unassigned_random(assignments, patients, df_rooms, df_surgeons, C_PER_SHIFT):
    """
    N3 (versão aleatória):
    - Escolhe UM paciente NÃO agendado ao acaso.
    - Escolhe UM bloco aleatório onde o cirurgião desse paciente está disponível
      e onde ele poderia caber se alguém saísse.
    - Escolhe UM paciente aleatório dentro do bloco para remover.
    - Realiza a troca APENAS se a capacidade permitir.
    - Apenas imprime quando um swap real acontece.
    """

    if assignments.empty:
        return assignments, False

    new_assign = assignments.copy()

    # pacientes agendados
    scheduled_ids = set(new_assign["patient_id"])

    # pacientes não agendados
    unassigned = patients[~patients["patient_id"].isin(scheduled_ids)]
    if unassigned.empty:
        return assignments, False

    # 1) escolher um NÃO agendado aleatório
    u_row = unassigned.sample(1).iloc[0]
    u_id = int(u_row["patient_id"])
    u_sid = int(u_row["surgeon_id"])
    need = int(u_row["duration"]) + CLEANUP

    # 2) blocos onde o cirurgião está disponível
    surg_ok = df_surgeons[
        (df_surgeons["surgeon_id"] == u_sid) &
        (df_surgeons["available"] == 1)
    ][["day", "shift"]]

    rooms_base = df_rooms[["room", "day", "shift", "available"]].copy()
    rooms_base["cap_min"] = rooms_base["available"] * C_PER_SHIFT

    used_by_block = new_assign.groupby(
        ["room", "day", "shift"], as_index=False
    ).agg(used_min=("used_min", "sum"))

    rooms_join = rooms_base.merge(
        used_by_block, on=["room", "day", "shift"], how="left"
    ).fillna({"used_min": 0})

    rooms_join["free_min"] = (rooms_join["cap_min"] - rooms_join["used_min"]).clip(lower=0)

    # 3) blocos candidatos
    cand_blocks = surg_ok.merge(
        rooms_join[rooms_join["available"]==1][["room","day","shift","used_min","cap_min","free_min"]],
        on=["day","shift"], how="inner"
    )

    if cand_blocks.empty:
        return assignments, False

    # 4) escolher bloco aleatório
    b = cand_blocks.sample(1).iloc[0]
    r, d, sh = int(b["room"]), int(b["day"]), int(b["shift"])

    # pacientes nesse bloco
    in_block = new_assign[
        (new_assign["room"] == r) &
        (new_assign["day"] == d) &
        (new_assign["shift"] == sh)
    ].merge(
        patients[["patient_id","priority","waiting","duration"]],
        on="patient_id", how="left"
    )

    if in_block.empty:
        return assignments, False

    # 5) paciente aleatório para sair
    out_row = in_block.sample(1).iloc[0]
    out_id = int(out_row["patient_id"])
    out_dur = int(out_row["duration"]) + CLEANUP

    # 6) verificar capacidade
    used_now = int(b["used_min"])
    cap = int(b["cap_min"])

    if used_now - out_dur + need > cap:
        return assignments, False

    # ============================
    #       PRINTS DO SWAP
    # ============================
    print("\n🔄 SWAP_WITH_UNASSIGNED_RANDOM:")
    print(f" → Inserir P{u_id} (prio={u_row['priority']}, wait={u_row['waiting']}, dur={u_row['duration']})")
    print(f" → Remover P{out_id} (prio={out_row['priority']}, wait={out_row['waiting']}, dur={out_row['duration']})")
    print(f" → Bloco escolhido: Room={r}, Day={d}, Shift={sh}\n")

    # 7) aplicar troca
    new_assign = new_assign[new_assign["patient_id"] != out_id].copy()

    new_assign = pd.concat([
        new_assign,
        pd.DataFrame([{
            "patient_id": u_id,
            "room": r,
            "day": d,
            "shift": sh,
            "used_min": need,
            "surgeon_id": u_sid,
            "iteration": new_assign["iteration"].max() + 1 if len(new_assign) else 1,
            "W_patient": None,
            "W_block": None
        }])
    ], ignore_index=True)

    new_assign = new_assign.sort_values(
        ["day", "shift", "room", "iteration"]
    ).reset_index(drop=True)

    print(" ✔️ Troca concluída!\n")

    return new_assign, True


def local_search(assign_init, df_rooms, df_surgeons, patients,
                 C_PER_SHIFT, max_no_improv=200):
    """
    Local search de first-improvement com duas vizinhanças discretas:
      - N₁: move_patient  → mover 1 doente para outro bloco viável
      - N₂: swap_patients → trocar blocos de 2 doentes

    Só mexe em (room, day, shift) dos doentes.
    """
    current = assign_init.copy()

    # avaliar solução inicial
    feas = feasibility_metrics(current, df_rooms, df_surgeons, patients, C_PER_SHIFT)
    best_feas = feas["feasibility_score"]
    best_rooms_join = feas["rooms_join"]

    if best_feas == 0:
        best_score = evaluate_schedule(current, patients, best_rooms_join)
    else:
        # penalizar soluções inviáveis
        best_score = -best_feas

    no_improv = 0

    while no_improv < max_no_improv:
        # escolher aleatoriamente o tipo de vizinho: N₁ (move) ou N₂ (swap)
        #move_type = random.choice(["move", "swap", "swap_unassigned"])
        move_type = "swap_with_unassigned_random"

        if move_type == "move":
            candidate, moved = move_patient(current, patients, df_rooms, df_surgeons, C_PER_SHIFT)
        elif move_type == "swap":
            candidate, moved = swap_patients(current, patients, df_rooms, df_surgeons, C_PER_SHIFT)
        elif move_type == 'swap_with_unassigned':
            candidate, moved = swap_with_unassigned(current, patients, df_rooms, df_surgeons, C_PER_SHIFT)
        elif move_type == 'swap_with_unassigned_random':
            candidate, moved = swap_with_unassigned_random(current, patients, df_rooms, df_surgeons, C_PER_SHIFT)
        
        
        if not moved:
            no_improv += 1
            continue
        else:
            print(f"\n--- Iter {no_improv}: tentando vizinhança {move_type}")
            print(f"Candidate: {candidate}")
        
        

        feas_c = feasibility_metrics(candidate, df_rooms, df_surgeons, patients, C_PER_SHIFT)
        rooms_join_c = feas_c["rooms_join"]

        if feas_c["feasibility_score"] == 0:
            cand_score = evaluate_schedule(candidate, patients, rooms_join_c)
        else:
            cand_score = -feas_c["feasibility_score"]
            
        print(f"cand_score = {cand_score:.4f}, best_score = {best_score:.4f}, feas = {feas_c['feasibility_score']}")


        if cand_score > best_score:
            print(" --> MELHORIA ENCONTRADA! (aceito)")
            current = candidate
            best_score = cand_score
            best_feas = feas_c["feasibility_score"]
            best_rooms_join = rooms_join_c
            no_improv = 0
        else:
            print(" --> não melhorou")
            no_improv += 1

    return current, best_score, best_feas

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
    print(f"Iter {iteration:02d}: "
          f"Assign P{int(patient_row['patient_id'])} → "
          f"(Room={int(best_block['room'])}, Day={int(best_block['day'])}, Shift={int(best_block['shift'])}), "
          f"W_patient={patient_row['W_patient']:.4f}, W_block={best_block['W_block']:.3f}")

print("\nFinal assignments:")

feas_init = feasibility_metrics(df_assignments, df_rooms, df_surgeons, df_patients, C_PER_SHIFT)
score_init = evaluate_schedule(df_assignments, df_patients, feas_init["rooms_join"])

print(f"Initial score = {score_init:.4f}, feasibility_score = {feas_init['feasibility_score']}")



print(df_assignments)

# ============================================
# LOCAL SEARCH em cima da solução construtiva
# ============================================
best_assignments, best_score, best_feas = local_search(
    df_assignments,
    df_rooms,
    df_surgeons,
    df_patients,
    C_PER_SHIFT,
    max_no_improv=200
)

print("\nAssignments depois da LOCAL SEARCH:")
print(best_assignments)
print(f"Score final = {best_score:.4f}, feasibility_score = {best_feas}")



# --------------------------------------------
# SURGEONS: remaining free minutes per day/shift
# --------------------------------------------
# base: one row per surgeon/day/shift with availability
df_surgeon_free = (
    df_surgeons[["surgeon_id", "day", "shift", "available"]]
    .drop_duplicates()
    .merge(df_surgeon_load[["surgeon_id", "day", "shift", "used_min"]],
           on=["surgeon_id", "day", "shift"], how="left")
    .fillna({"used_min": 0})
)

# capacity is C_PER_SHIFT only if surgeon is available in that block
df_surgeon_free["cap_min"]  = df_surgeon_free["available"] * C_PER_SHIFT
df_surgeon_free["free_min"] = (df_surgeon_free["cap_min"] - df_surgeon_free["used_min"]).clip(lower=0)

# utilization guard (avoid division by zero)
df_surgeon_free["utilization"] = df_surgeon_free.apply(
    lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0.0, axis=1
)

df_surgeon_free = df_surgeon_free.sort_values(["surgeon_id", "day", "shift"]).reset_index(drop=True)

print("\nSurgeons — remaining free minutes per day/shift:")
print(df_surgeon_free.head(12))


# --------------------------------------------
# ROOMS: remaining free minutes per day/shift
# --------------------------------------------
# df_capacity already holds current free_min per (room,day,shift)
df_room_free = df_capacity[["room", "day", "shift", "available", "free_min"]].copy()

# derive used minutes and utilization
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
    df_patients[["patient_id", "surgeon_id", "duration", "priority", "waiting"]],
    on="patient_id", how="left"
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
print(f"Initial score = {score_init:.4f}, feasibility_score = {feas_init['feasibility_score']}")

