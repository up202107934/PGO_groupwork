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
import warnings
import time

# Suprimir todos os warnings
warnings.filterwarnings("ignore")

# ------------------------------
# PARAMETERS
# ------------------------------
DATA_FILE = "Instance_C2_30.dat"

C_PER_SHIFT = 360   # minutes per shift (6h * 60)
CLEANUP = 17        # cleaning time

ALPHA1 = 0.25  # priority
ALPHA2 = 0.25  # waited days
ALPHA3 = 0.25  # deadline closeness
ALPHA4 = 0.25  # feasible blocks

TOLERANCE = 15  # NEW- tolerance minutes to pass the capacity of the shift
MAX_TIME_PER_MOVE = 0.5 * 60  # Tempo máximo por move em segundos (1 minuto)


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

    # score total de infeasibilidade (minimizar; 0 => solução "ideal")
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

def evaluate_schedule(assignments, patients, rooms_free, excess_block_min,
                      weights=(0.8, 0.05, 0.05, 0.1)):
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
        # 2) WAITING TERM (relativo ao deadline de cada prioridade)
        # ------------------------------
        # Calcula para cada paciente: waiting / deadline_limit
        # Pacientes mais próximos do deadline (proporcionalmente) têm score maior
        merged["deadline_limit"] = merged["priority"].apply(deadline_limit_from_priority)
        merged["wait_ratio"] = merged.apply(
            lambda r: r["waiting"] / r["deadline_limit"] if r["deadline_limit"] > 0 else 0.0,
            axis=1
        )
        norm_wait_term = float(merged["wait_ratio"].mean())

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
        - 0.001 * excess_block_min
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

    # 5) Adicionar coluna continuity (para compatibilidade com score_block_for_patient)
    if len(assignments) and len(cand) > 0:
        surg_prev = assignments[assignments["surgeon_id"] == sid][["day","shift","room"]]
        cand = cand.merge(
            surg_prev.assign(continuity=1).drop_duplicates(subset=["day","shift","room"]),
            on=["day","shift","room"],
            how="left"
        ).fillna({"continuity": 0})
    else:
        cand["continuity"] = 0

    return cand


def generate_neighbor_add_only(current_assignments,
                               df_patients,
                               df_rooms,
                               df_surgeons,
                               C_PER_SHIFT,
                               max_add=2):
    """Tenta apenas INSERIR pacientes não agendados (sem remover ninguém).

    Retorna um novo assignments e a lista de pacientes realmente adicionados.
    """
    assignments = current_assignments.copy()

    scheduled_ids = set(assignments["patient_id"].unique().tolist())
    unassigned_ids = [
        pid for pid in df_patients["patient_id"].unique().tolist()
        if pid not in scheduled_ids
    ]

    if not unassigned_ids:
        return assignments, []

    random.shuffle(unassigned_ids)
    ids_in_candidates = unassigned_ids[:max_add]
    ids_in_effective = []

    for pid in ids_in_candidates:
        prow = df_patients[df_patients["patient_id"] == pid]
        if prow.empty:
            continue
        prow = prow.iloc[0]

        cand_blocks = candidate_blocks_for_patient_in_solution(
            assignments, prow, df_rooms, df_surgeons, C_PER_SHIFT
        )

        if len(cand_blocks) == 0:
            continue

        chosen = cand_blocks.sort_values(["day", "shift", "room"]).iloc[0]
        need = int(prow["duration"]) + CLEANUP

        new_row = {
            "patient_id": int(pid),
            "room": int(chosen["room"]),
            "day": int(chosen["day"]),
            "shift": int(chosen["shift"]),
            "used_min": need,
            "surgeon_id": int(prow["surgeon_id"]),
            "iteration": -1,
            "W_patient": None,
            "W_block": None,
        }

        assignments = pd.concat(
            [assignments, pd.DataFrame([new_row])],
            ignore_index=True
        )

        ids_in_effective.append(int(pid))

    return assignments, ids_in_effective


def generate_neighbor_intra_surgeon_swap(current_assignments,
                                          df_patients,
                                          df_rooms,
                                          df_surgeons,
                                          C_PER_SHIFT):
    """
    Troca pacientes do MESMO cirurgião entre blocos diferentes.
    Tipos de troca: 1-1, 1-2, ou 2-1
    
    Retorna:
      - novo assignments
      - info da troca (surgeon_id, blocos, pacientes trocados)
      - sucesso (True/False)
    """
    if current_assignments.empty or len(current_assignments) < 2:
        return current_assignments, {}, False
    
    assignments = current_assignments.copy()
    
    # Agrupar por cirurgião e bloco
    assignments['block_key'] = assignments.apply(
        lambda r: (int(r['room']), int(r['day']), int(r['shift'])), axis=1
    )
    
    # Cirurgiões que operam em pelo menos 2 blocos diferentes
    surgeon_blocks = assignments.groupby('surgeon_id')['block_key'].nunique()
    eligible_surgeons = surgeon_blocks[surgeon_blocks >= 2].index.tolist()
    
    if not eligible_surgeons:
        return current_assignments, {}, False
    
    # Escolher cirurgião aleatório
    surgeon_id = random.choice(eligible_surgeons)
    
    # Pacientes deste cirurgião
    surg_patients = assignments[assignments['surgeon_id'] == surgeon_id].copy()
    
    # Blocos diferentes onde opera
    blocks = surg_patients['block_key'].unique().tolist()
    
    if len(blocks) < 2:
        return current_assignments, {}, False
    
    # Escolher 2 blocos diferentes
    block1, block2 = random.sample(blocks, 2)
    
    # Pacientes em cada bloco
    patients_b1 = surg_patients[surg_patients['block_key'] == block1]['patient_id'].tolist()
    patients_b2 = surg_patients[surg_patients['block_key'] == block2]['patient_id'].tolist()
    
    if not patients_b1 or not patients_b2:
        return current_assignments, {}, False
    
    # Escolher tipo de swap aleatoriamente (1-1, 1-2, 2-1)
    swap_type = random.choice(['1-1','1-2','2-1'])
    
    # Selecionar pacientes
    if swap_type == '1-1':
        if len(patients_b1) < 1 or len(patients_b2) < 1:
            return current_assignments, {}, False
        ids_from_b1 = random.sample(patients_b1, 1)
        ids_from_b2 = random.sample(patients_b2, 1)
    elif swap_type == '1-2':
        if len(patients_b1) < 1 or len(patients_b2) < 2:
            return current_assignments, {}, False
        ids_from_b1 = random.sample(patients_b1, 1)
        ids_from_b2 = random.sample(patients_b2, 2)
    else:  # 2-1
        if len(patients_b1) < 2 or len(patients_b2) < 1:
            return current_assignments, {}, False
        ids_from_b1 = random.sample(patients_b1, 2)
        ids_from_b2 = random.sample(patients_b2, 1)
    
    # Calcular tempo necessário
    def get_total_duration(patient_ids):
        total = 0
        for pid in patient_ids:
            p_data = df_patients[df_patients['patient_id'] == pid]
            if not p_data.empty:
                total += int(p_data['duration'].iloc[0]) + CLEANUP
        return total
    
    time_b1_out = get_total_duration(ids_from_b1)
    time_b2_out = get_total_duration(ids_from_b2)
    time_b1_in = time_b2_out
    time_b2_in = time_b1_out
    
    # Verificar capacidade em cada bloco
    rooms_base = df_rooms[["room", "day", "shift", "available"]].copy()
    rooms_base["cap_min"] = rooms_base["available"] * C_PER_SHIFT
    
    used_by_block = assignments.groupby(["room", "day", "shift"], as_index=False).agg(used_min=("used_min", "sum"))
    rooms_join = rooms_base.merge(used_by_block, on=["room", "day", "shift"], how="left").fillna({"used_min": 0})
    
    # Capacidade bloco 1
    r1, d1, s1 = block1
    block1_data = rooms_join[(rooms_join['room'] == r1) & 
                             (rooms_join['day'] == d1) & 
                             (rooms_join['shift'] == s1)]
    if block1_data.empty:
        return current_assignments, {}, False
    cap_b1 = block1_data['cap_min'].iloc[0]
    used_b1 = block1_data['used_min'].iloc[0]
    
    # Capacidade bloco 2
    r2, d2, s2 = block2
    block2_data = rooms_join[(rooms_join['room'] == r2) & 
                             (rooms_join['day'] == d2) & 
                             (rooms_join['shift'] == s2)]
    if block2_data.empty:
        return current_assignments, {}, False
    cap_b2 = block2_data['cap_min'].iloc[0]
    used_b2 = block2_data['used_min'].iloc[0]
    
    # Verificar se swap cabe
    new_used_b1 = used_b1 - time_b1_out + time_b1_in
    new_used_b2 = used_b2 - time_b2_out + time_b2_in
    
    if new_used_b1 > cap_b1 + TOLERANCE or new_used_b2 > cap_b2 + TOLERANCE:
        return current_assignments, {}, False
    
    # Aplicar swap
    new_assign = assignments.copy()
    
    # Trocar pacientes de b1 para b2
    for pid in ids_from_b1:
        idx = new_assign[new_assign['patient_id'] == pid].index
        if len(idx) == 0:
            continue
        new_assign.loc[idx, ['room', 'day', 'shift']] = [r2, d2, s2]
        # Atualizar used_min
        p_data = df_patients[df_patients['patient_id'] == pid]
        if not p_data.empty:
            dur = int(p_data['duration'].iloc[0]) + CLEANUP
            new_assign.loc[idx, 'used_min'] = dur
    
    # Trocar pacientes de b2 para b1
    for pid in ids_from_b2:
        idx = new_assign[new_assign['patient_id'] == pid].index
        if len(idx) == 0:
            continue
        new_assign.loc[idx, ['room', 'day', 'shift']] = [r1, d1, s1]
        # Atualizar used_min
        p_data = df_patients[df_patients['patient_id'] == pid]
        if not p_data.empty:
            dur = int(p_data['duration'].iloc[0]) + CLEANUP
            new_assign.loc[idx, 'used_min'] = dur
    
    # Remover coluna auxiliar
    new_assign = new_assign.drop(columns=['block_key'])
    new_assign = new_assign.sort_values(['day', 'shift', 'room', 'iteration']).reset_index(drop=True)
    
    swap_info = {
        'surgeon_id': int(surgeon_id),
        'swap_type': swap_type,
        'block1': block1,
        'block2': block2,
        'from_b1': ids_from_b1,
        'from_b2': ids_from_b2
    }
    
    return new_assign, swap_info, True


def generate_neighbor_swap(current_assignments,
                           df_patients,
                           df_rooms,
                           df_surgeons,
                           C_PER_SHIFT,
                           max_swap_out=2,
                           max_swap_in=2):
    """
    Gera um vizinho do tipo k-out / l-in:
      - remove até max_swap_out pacientes agendados
      - tenta inserir até max_swap_in pacientes não agendados
        em blocos viáveis (usando candidate_blocks_for_patient_in_solution).
    Devolve:
      - assignments vizinho
      - lista de ids removidos
      - lista de ids efectivamente adicionados
    """
    assignments = current_assignments.copy()

    # pacientes actualmente agendados
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

        # blocos viáveis dado o assignments ACTUAL
        cand_blocks = candidate_blocks_for_patient_in_solution(
            assignments,
            prow,
            df_rooms,
            df_surgeons,
            C_PER_SHIFT
        )

        if cand_blocks.empty:
            continue

        # escolhe um bloco (p.ex. mais cedo)
        chosen = cand_blocks.sort_values(["day", "shift", "room"]).iloc[0]

        need = int(prow["duration"]) + CLEANUP
        sid = int(prow["surgeon_id"])

        new_row = {
            "patient_id": int(pid),
            "room": int(chosen["room"]),
            "day": int(chosen["day"]),
            "shift": int(chosen["shift"]),
            "used_min": need,
            "surgeon_id": sid,
            "iteration": -1,      # marca que veio da ILS
            "W_patient": None,
            "W_block": None,
        }

        assignments = pd.concat(
            [assignments, pd.DataFrame([new_row])],
            ignore_index=True,
            sort=False
        )

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
    print(f"Iter {iteration:02d}: "
          f"Assign P{int(patient_row['patient_id'])} → "
          f"(Room={int(best_block['room'])}, Day={int(best_block['day'])}, Shift={int(best_block['shift'])}), "
          f"W_patient={patient_row['W_patient']:.4f}, W_block={best_block['W_block']:.3f}")

#print("\nFinal assignments:")

feas_init = feasibility_metrics(df_assignments, df_rooms, df_surgeons, df_patients, C_PER_SHIFT)

rooms_eval = feas_init["rooms_cap_join"].copy()
rooms_eval["utilization"] = rooms_eval.apply(
    lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0,
    axis=1
)

# Primeiro enriquecer assignments com priority, waiting e duration
assign_eval = df_assignments.merge(
    df_patients[["patient_id", "priority", "waiting", "duration"]],
    on="patient_id",
    how="left"
)

# depois calcular score
score_init = evaluate_schedule(
    assign_eval,
    df_patients,
    rooms_eval,
    feas_init["excess_block_min"]
)


print(f"Initial score = {score_init['score']:.4f}, ")

# ============================================================
#     LOGGING INFRASTRUCTURE FOR SENSITIVITY ANALYSIS
# ============================================================

improvement_log = []
iteration_log = []  # Para LS1, LS2, LS3
ils_log = []  # Para ILS/VNS

def log_iteration(fase, iteracao, metrics, accepted,
                  ids_out=None, ids_in=None):
    """
    Regista as métricas de TODAS as iterações para análise de sensibilidade.
    """
    iteration_log.append({
        "fase": fase,
        "iteracao": int(iteracao),
        "score": float(metrics["score"]),
        "n_patient": int(metrics["assigned_patients"]),
        "scheduled_rooms_r": float(metrics["ratio_scheduled_raw"]),
        "u_waiting_k": float(metrics["avg_waiting_raw"]),
        "p_priority_j": float(metrics["avg_priority_raw"]),
        "w_overdue_l": int(metrics["deadline_overdue_patients"]),
        "f_block_m": int(metrics["excess_block_min_raw"]),
        "surgeon_min_raw": int(metrics["excess_surgeon_min_raw"]),
        "acceptable": int(bool(accepted)),

        # NOVAS COLUNAS
        "ids_out": ",".join(map(str, ids_out)) if ids_out else "",
        "ids_in": ",".join(map(str, ids_in)) if ids_in else "",
    })



def log_ils_iteration(fase, iteracao, metrics, ils_iteration=None, ls_phase=None, accepted=None, shaking_id=None):
    """
    Regista as métricas do ILS/VNS (tanto shakings como LS internas).
    """
    log_entry = {
        "fase": fase,
        "iteracao": iteracao if not isinstance(iteracao, int) else int(iteracao),  # Pode ser "S1", "S2" ou número
        "score": float(metrics["score"]),
        "n_patient": int(metrics["assigned_patients"]),
        "scheduled_rooms_r": float(metrics["ratio_scheduled_raw"]),
        "u_waiting_k": float(metrics["avg_waiting_raw"]),
        "p_priority_j": float(metrics["avg_priority_raw"]),
        "w_overdue_l": int(metrics["deadline_overdue_patients"]),
        "f_block_m": int(metrics["excess_block_min_raw"]),
        "surgeon_min_raw": int(metrics["excess_surgeon_min_raw"]),
    }
    
    # Adicionar campos opcionais se fornecidos
    if shaking_id is not None:
        log_entry["shaking_id"] = shaking_id
    if ils_iteration is not None:
        log_entry["ils_iteration"] = int(ils_iteration)
    if ls_phase is not None:
        log_entry["ls_phase"] = ls_phase
    if accepted is not None:
        log_entry["acceptable"] = int(bool(accepted))  # Última coluna
    
    ils_log.append(log_entry)


def eval_components(assignments_df, rooms_free_df, feas_dict):
    """
    Calcula métricas detalhadas para o iteration_log.
    """
    total_patients = len(df_patients)

    # Pacientes agendados
    assigned_patients = int(assignments_df["patient_id"].nunique()) if len(assignments_df) else 0
    ratio_scheduled_raw = assigned_patients / total_patients if total_patients > 0 else 0.0

    # Utilização das salas
    util_rooms_raw = float(
        rooms_free_df.loc[rooms_free_df["cap_min"] > 0, "utilization"].mean()
    ) if len(rooms_free_df) else 0.0

    # Priority e Waiting médios
    if len(assignments_df):
        # Garantir que temos as colunas priority e waiting
        if "priority" not in assignments_df.columns or "waiting" not in assignments_df.columns:
            merged = assignments_df.merge(
                df_patients[["patient_id", "priority", "waiting"]],
                on="patient_id", how="left"
            )
        else:
            merged = assignments_df

        avg_priority_raw = float(merged["priority"].mean())
        avg_waiting_raw = float(merged["waiting"].mean())
    else:
        avg_priority_raw = 0.0
        avg_waiting_raw = 0.0

    # Deadline overdues APENAS entre pacientes agendados
    if len(assignments_df):
        cols_needed = ["priority", "waiting"]
        missing_cols = [c for c in cols_needed if c not in assignments_df.columns]

        if missing_cols:
            pats_assigned = assignments_df.merge(
                df_patients[["patient_id", "priority", "waiting"]],
                on="patient_id",
                how="left"
            )
        else:
            pats_assigned = assignments_df.copy()

        pats_assigned["deadline_limit"] = pats_assigned["priority"].apply(deadline_limit_from_priority)
        pats_assigned["overdue_days"] = (pats_assigned["waiting"] - pats_assigned["deadline_limit"]).clip(lower=0)
        deadline_overdue_patients = int((pats_assigned["overdue_days"] > 0).sum())
    else:
        deadline_overdue_patients = 0

    # Excessos de bloco e cirurgião
    excess_block_min_raw = int(feas_dict["excess_block_min"])
    excess_surgeon_min_raw = int(feas_dict["excess_surgeon_min"])

    # Avaliação do score
    ev = evaluate_schedule(
        assignments=assignments_df,
        patients=df_patients,
        rooms_free=rooms_free_df,
        excess_block_min=feas_dict["excess_block_min"]
    )

    return {
        "score": float(ev["score"]),
        "assigned_patients": assigned_patients,
        "ratio_scheduled_raw": ratio_scheduled_raw,
        "util_rooms_raw": util_rooms_raw,
        "avg_waiting_raw": avg_waiting_raw,
        "avg_priority_raw": avg_priority_raw,
        "deadline_overdue_patients": deadline_overdue_patients,
        "excess_block_min_raw": excess_block_min_raw,
        "excess_surgeon_min_raw": excess_surgeon_min_raw,
    }


# =========================================================
#       LS #1: SWAP i-j
# =========================================================

N_ILS_ITER = 100
MAX_SWAP_OUT = 2
MAX_SWAP_IN  = 2

# ponto de partida: solução construtiva
current_assignments = df_assignments.copy()

# avaliação inicial
feas_init = feasibility_metrics(current_assignments, df_rooms, df_surgeons, df_patients, C_PER_SHIFT)
rooms_init = feas_init["rooms_cap_join"].copy()
rooms_init["utilization"] = rooms_init.apply(
    lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0,
    axis=1
)
assign_init = current_assignments.merge(
    df_patients[["patient_id", "priority", "waiting", "duration"]],
    on="patient_id",
    how="left"
)
current_score = evaluate_schedule(
    assign_init,
    df_patients,
    rooms_init,
    feas_init["excess_block_min"]
)["score"]
print(f"Initial score: {current_score:.4f}")

best_score = current_score
best_assignments = current_assignments.copy()
best_rooms_free = rooms_init.copy()
best_feas = feas_init

print("\n========== LS #1: SWAP i-j ==========")


ls1_start_time = time.time()
it = 0

while (time.time() - ls1_start_time) < MAX_TIME_PER_MOVE:
    # -------- 1) MOVE --------
    neighbor_struct, ids_out, ids_in = generate_neighbor_swap(
        current_assignments,
        df_patients,
        df_rooms,
        df_surgeons,
        C_PER_SHIFT,
        max_swap_out=MAX_SWAP_OUT,
        max_swap_in=MAX_SWAP_IN
    )

    # -------- 2) AVALIAÇÃO --------
    feas_n = feasibility_metrics(neighbor_struct, df_rooms, df_surgeons, df_patients, C_PER_SHIFT)
    rooms_n = feas_n["rooms_cap_join"].copy()
    rooms_n["utilization"] = rooms_n.apply(
        lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0,
        axis=1
    )
    neighbor_enriched = neighbor_struct.merge(
        df_patients[["patient_id", "priority", "waiting", "duration"]],
        on="patient_id",
        how="left"
    )
    neigh_score = evaluate_schedule(
        neighbor_enriched,
        df_patients,
        rooms_n,
        feas_n["excess_block_min"]
    )["score"]

    # Log da iteração
    new_metrics = eval_components(neighbor_enriched, rooms_n, feas_n)
    log_iteration(
        fase="LS1_SWAP",
        iteracao=it,
        metrics=new_metrics,
        accepted=(neigh_score > current_score),
        ids_out=ids_out,
        ids_in=ids_in
    )


    if neigh_score > current_score:
        current_assignments = neighbor_struct.copy()
        current_score = neigh_score

        if neigh_score > best_score:
            best_score = neigh_score
            best_assignments = neighbor_struct.copy()
            best_rooms_free = rooms_n.copy()
            best_feas = feas_n
            print(f"[LS1 Iter {it}] Improved to {neigh_score:.4f} | removed={ids_out} | added={ids_in}")
    
    it += 1

print(f"\nLS #1 final score = {best_score:.4f}")

# ============================================================
#     LS #2 — INTRA-SURGEON SWAP
# ============================================================

print("\n========== LS #2: INTRA-SURGEON SWAP ==========")

current_assignments = best_assignments.copy()
current_score = best_score

N_LS2_ITER = 100

print(f"Initial score: {current_score:.4f}")

ls2_start_time = time.time()
it = 0

while (time.time() - ls2_start_time) < MAX_TIME_PER_MOVE:
    neighbor, swap_info, success = generate_neighbor_intra_surgeon_swap(
        current_assignments, df_patients, df_rooms, df_surgeons, C_PER_SHIFT
    )
    
    if not success:
        continue
    
    # ---- avaliação ----
    feas_n = feasibility_metrics(neighbor, df_rooms, df_surgeons, df_patients, C_PER_SHIFT)
    rooms_n = feas_n["rooms_cap_join"].copy()
    rooms_n["utilization"] = rooms_n.apply(
        lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0,
        axis=1
    )
    neighbor_enriched = neighbor.merge(
        df_patients[["patient_id", "priority", "waiting", "duration"]],
        on="patient_id",
        how="left"
    )
    neigh_score = evaluate_schedule(
        neighbor_enriched,
        df_patients,
        rooms_n,
        feas_n["excess_block_min"]
    )["score"]
    
    # Log da iteração LS2
    new_metrics = eval_components(neighbor_enriched, rooms_n, feas_n)
    log_iteration("LS2_INTRA_SURGEON_SWAP", it, new_metrics, accepted=(neigh_score > current_score))
    
    if neigh_score > current_score:
        current_assignments = neighbor.copy()
        current_score = neigh_score
        
        if neigh_score > best_score:
            best_score = neigh_score
            best_assignments = neighbor.copy()
            best_rooms_free = rooms_n.copy()
            best_feas = feas_n
            print(f"[LS2 Iter {it}] Improved to {current_score:.4f} | "
                  f"surgeon={swap_info['surgeon_id']}, type={swap_info['swap_type']}")
    
    it += 1

print(f"\nLS #2 final score = {best_score:.4f}")

# ============================================================
#     LS #3 — ADD-ONLY
# ============================================================

print("\n========== LS #3: ADD-ONLY ==========")

current_assignments = best_assignments.copy()
current_score = best_score

N_LS3_ITER = 500

print(f"Initial add-only score: {current_score:.4f}")

ls3_start_time = time.time()
it = 0

while (time.time() - ls3_start_time) < MAX_TIME_PER_MOVE:
    neighbor, ids_added = generate_neighbor_add_only(
        current_assignments, df_patients, df_rooms, df_surgeons, C_PER_SHIFT, max_add=2
    )
    
    # Se não conseguiu adicionar nenhum paciente, não há mais vizinhos viáveis
    if not ids_added:
        print(f"[LS3] No more patients can be added. Stopping after {it} iterations.")
        break

    # ---- avaliação ----
    feas_n = feasibility_metrics(neighbor, df_rooms, df_surgeons, df_patients, C_PER_SHIFT)
    rooms_n = feas_n["rooms_cap_join"].copy()
    rooms_n["utilization"] = rooms_n.apply(
        lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0,
        axis=1
    )
    neighbor_enriched = neighbor.merge(
        df_patients[["patient_id", "priority", "waiting", "duration"]],
        on="patient_id",
        how="left"
    )
    neigh_score = evaluate_schedule(
        neighbor_enriched,
        df_patients,
        rooms_n,
        feas_n["excess_block_min"]
    )["score"]

    # Log da iteração LS3
    new_metrics = eval_components(neighbor_enriched, rooms_n, feas_n)
    log_iteration(
        "LS3_ADD_ONLY",
        it,
        new_metrics,
        accepted=(neigh_score > current_score),
        ids_out=[],
        ids_in=ids_added
    )

    if neigh_score > current_score:
        current_assignments = neighbor.copy()
        current_score = neigh_score

        if neigh_score > best_score:
            best_score = neigh_score
            best_assignments = neighbor.copy()
            best_rooms_free = rooms_n.copy()
            best_feas = feas_n
            print(f"[LS3 Iter {it}] Improved score to {current_score:.4f} | added={ids_added}")
    
    it += 1

print(f"\nLS #3 final score = {best_score:.4f}\n")

# ============================================================
#     ILS/VNS — Iterated Local Search com Variable Neighborhood
# ============================================================
import time

def run_local_search_phase(start_assignments, df_patients, df_rooms, df_surgeons, 
                           C_PER_SHIFT, verbose=False, 
                           shaking_info=None, ils_iteration=None, shaking_id=None):
    """
    Executa as 3 fases de Local Search (SWAP, INTRA-SURGEON, ADD-ONLY)
    a partir de uma solução inicial. Retorna a melhor solução encontrada.
    """
    current_assignments = start_assignments.copy()
    
    # Contadores separados para cada fase LS
    ls1_improvement_count = 0
    ls2_improvement_count = 0
    ls3_improvement_count = 0
    
    # Avaliação inicial
    feas = feasibility_metrics(current_assignments, df_rooms, df_surgeons, df_patients, C_PER_SHIFT)
    rooms_eval = feas["rooms_cap_join"].copy()
    rooms_eval["utilization"] = rooms_eval.apply(
        lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0, axis=1
    )
    assign_eval = current_assignments.merge(
        df_patients[["patient_id", "priority", "waiting", "duration"]],
        on="patient_id", how="left"
    )
    current_score = evaluate_schedule(assign_eval, df_patients, rooms_eval, feas["excess_block_min"])["score"]
    
    best_ls_score = current_score
    best_ls_assignments = current_assignments.copy()
    best_ls_feas = feas
    best_ls_rooms = rooms_eval.copy()
    
    if verbose and shaking_info:
        print(f"\n      [ILS Iter {shaking_info['iteration']}] {shaking_info['name']}")
        print(f"      → Shaking score: {current_score:.4f} (removed={shaking_info['removed']}, added={shaking_info['added']})")
    elif verbose:
        print(f"      → Shaking score: {current_score:.4f}")
    
    # --- LS Phase 1: ADD-ONLY ---
    if verbose:
        print(f"\n      ========== LS #1: ADD-ONLY ==========")
        print(f"      Initial add-only score: {current_score:.4f}")
    
    ls1_phase_start = time.time()
    ls1_iter = 0
    while (time.time() - ls1_phase_start) < MAX_TIME_PER_MOVE:
        ls1_iter += 1
        neighbor, ids_added = generate_neighbor_add_only(
            current_assignments, df_patients, df_rooms, df_surgeons, C_PER_SHIFT, max_add=2
        )
        if not ids_added:
            if verbose:
                print(f"      [LS1] No more patients can be added. Stopping after {ls1_iter-1} iterations.")
            break
        feas_n = feasibility_metrics(neighbor, df_rooms, df_surgeons, df_patients, C_PER_SHIFT)
        rooms_n = feas_n["rooms_cap_join"].copy()
        rooms_n["utilization"] = rooms_n.apply(
            lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0, axis=1
        )
        neighbor_enriched = neighbor.merge(
            df_patients[["patient_id", "priority", "waiting", "duration"]],
            on="patient_id", how="left"
        )
        neigh_score = evaluate_schedule(neighbor_enriched, df_patients, rooms_n, feas_n["excess_block_min"])["score"]
        
        # Log TODAS as iterações LS1 dentro do ILS
        if ils_iteration is not None:
            ls_metrics = eval_components(neighbor_enriched, rooms_n, feas_n)
            log_ils_iteration(
                fase="ILS_LS1_ADD_ONLY",
                iteracao=ls1_iter,
                metrics=ls_metrics,
                ils_iteration=ils_iteration,
                ls_phase="LS1_ADD_ONLY",
                accepted=(neigh_score > current_score),
                shaking_id=shaking_id
            )
        
        if neigh_score > current_score:
            current_assignments = neighbor.copy()
            current_score = neigh_score
            ls1_improvement_count += 1
            
            if verbose:
                print(f"      [LS1 Iter {ls1_iter}] Improved score to {neigh_score:.4f} | added={ids_added}")
            
            if neigh_score > best_ls_score:
                best_ls_score = neigh_score
                best_ls_assignments = neighbor.copy()
                best_ls_feas = feas_n
                best_ls_rooms = rooms_n.copy()
    
    if verbose:
        print(f"\n      LS #1 final score = {best_ls_score:.4f}")
    
    # --- LS Phase 2: INTRA-SURGEON SWAP ---
    if verbose:
        print(f"\n      ========== LS #2: INTRA-SURGEON SWAP ==========")
        print(f"      Initial score: {best_ls_score:.4f}")
    
    current_assignments = best_ls_assignments.copy()
    current_score = best_ls_score
    ls2_phase_start = time.time()
    ls2_iter = 0
    while (time.time() - ls2_phase_start) < MAX_TIME_PER_MOVE:
        ls2_iter += 1
        neighbor, swap_info, success = generate_neighbor_intra_surgeon_swap(
            current_assignments, df_patients, df_rooms, df_surgeons, C_PER_SHIFT
        )
        if not success:
            continue
        feas_n = feasibility_metrics(neighbor, df_rooms, df_surgeons, df_patients, C_PER_SHIFT)
        rooms_n = feas_n["rooms_cap_join"].copy()
        rooms_n["utilization"] = rooms_n.apply(
            lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0, axis=1
        )
        neighbor_enriched = neighbor.merge(
            df_patients[["patient_id", "priority", "waiting", "duration"]],
            on="patient_id", how="left"
        )
        neigh_score = evaluate_schedule(neighbor_enriched, df_patients, rooms_n, feas_n["excess_block_min"])["score"]
        
        # Log TODAS as iterações LS2 dentro do ILS
        if ils_iteration is not None:
            ls_metrics = eval_components(neighbor_enriched, rooms_n, feas_n)
            log_ils_iteration(
                fase="ILS_LS2_INTRA_SURGEON",
                iteracao=ls2_iter,
                metrics=ls_metrics,
                ils_iteration=ils_iteration,
                ls_phase="LS2_INTRA_SURGEON",
                accepted=(neigh_score > current_score),
                shaking_id=shaking_id
            )
        
        if neigh_score > current_score:
            current_assignments = neighbor.copy()
            current_score = neigh_score
            ls2_improvement_count += 1
            
            if verbose:
                print(f"      [LS2 Iter {ls2_iter}] Improved to {neigh_score:.4f} | "
                      f"surgeon={swap_info['surgeon_id']}, type={swap_info['swap_type']}")
            
            if neigh_score > best_ls_score:
                best_ls_score = neigh_score
                best_ls_assignments = neighbor.copy()
                best_ls_feas = feas_n
                best_ls_rooms = rooms_n.copy()
    
    if verbose:
        print(f"\n      LS #2 final score = {best_ls_score:.4f}")
    
    # --- LS Phase 3: SWAP i-j ---
    if verbose:
        print(f"\n      ========== LS #3: SWAP i-j ==========")
        print(f"      Initial score: {best_ls_score:.4f}")
    
    current_assignments = best_ls_assignments.copy()
    current_score = best_ls_score
    ls3_phase_start = time.time()
    ls3_iter = 0
    while (time.time() - ls3_phase_start) < MAX_TIME_PER_MOVE:
        ls3_iter += 1
        neighbor, ids_out, ids_in = generate_neighbor_swap(
            current_assignments, df_patients, df_rooms, df_surgeons, C_PER_SHIFT,
            max_swap_out=2, max_swap_in=2
        )
        feas_n = feasibility_metrics(neighbor, df_rooms, df_surgeons, df_patients, C_PER_SHIFT)
        rooms_n = feas_n["rooms_cap_join"].copy()
        rooms_n["utilization"] = rooms_n.apply(
            lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0, axis=1
        )
        neighbor_enriched = neighbor.merge(
            df_patients[["patient_id", "priority", "waiting", "duration"]],
            on="patient_id", how="left"
        )
        neigh_score = evaluate_schedule(neighbor_enriched, df_patients, rooms_n, feas_n["excess_block_min"])["score"]
        
        # Log TODAS as iterações LS3 dentro do ILS
        if ils_iteration is not None:
            ls_metrics = eval_components(neighbor_enriched, rooms_n, feas_n)
            log_ils_iteration(
                fase="ILS_LS3_SWAP_IJ",
                iteracao=ls3_iter,
                metrics=ls_metrics,
                ils_iteration=ils_iteration,
                ls_phase="LS3_SWAP_IJ",
                accepted=(neigh_score > current_score),
                shaking_id=shaking_id
            )
        
        if neigh_score > current_score:
            current_assignments = neighbor.copy()
            current_score = neigh_score
            ls3_improvement_count += 1
            
            if verbose:
                print(f"      [LS3 Iter {ls3_iter}] Improved to {neigh_score:.4f} | removed={ids_out} | added={ids_in}")
            
            if neigh_score > best_ls_score:
                best_ls_score = neigh_score
                best_ls_assignments = neighbor.copy()
                best_ls_feas = feas_n
                best_ls_rooms = rooms_n.copy()
    
    if verbose:
        print(f"\n      LS #3 final score = {best_ls_score:.4f}")
    
    return best_ls_assignments, best_ls_score, best_ls_feas, best_ls_rooms


def shaking(assignments, df_patients, df_rooms, df_surgeons, C_PER_SHIFT,
            min_out, max_out, min_in, max_in):
    """
    Perturbação da solução: remove entre min_out e max_out pacientes,
    tenta inserir entre min_in e max_in pacientes não agendados.
    Retorna (result, ids_out, ids_in_effective, success)
    """
    result = assignments.copy()
    
    # Número de remoções e inserções
    scheduled_ids = result["patient_id"].unique().tolist()
    all_ids = df_patients["patient_id"].unique().tolist()
    unassigned_ids = [pid for pid in all_ids if pid not in scheduled_ids]
    
    # Verificar se há pacientes suficientes para o shaking
    if len(scheduled_ids) < min_out:
        # Não há pacientes agendados suficientes para remover
        return assignments, [], [], False
    
    # Determinar quantos remover (entre min_out e max_out, limitado aos disponíveis)
    k_out = random.randint(min_out, min(max_out, len(scheduled_ids)))
    
    # Remover k_out pacientes aleatórios
    ids_out = random.sample(scheduled_ids, k_out)
    result = result[~result["patient_id"].isin(ids_out)].copy()
    
    # Determinar quantos inserir (flexível: se não há suficientes, insere o que for possível)
    if len(unassigned_ids) == 0:
        # Não há pacientes não agendados - shaking só remove
        return result, ids_out, [], True
    
    k_in = random.randint(min(min_in, len(unassigned_ids)), min(max_in, len(unassigned_ids)))
    
    # Tentar inserir k_in pacientes
    ids_in_effective = []
    random.shuffle(unassigned_ids)
    candidates_in = unassigned_ids[:k_in]
    
    for pid in candidates_in:
        prow = df_patients[df_patients["patient_id"] == pid]
        if prow.empty:
            continue
        prow = prow.iloc[0]
        
        cand_blocks = candidate_blocks_for_patient_in_solution(
            result, prow, df_rooms, df_surgeons, C_PER_SHIFT
        )
        
        if len(cand_blocks) == 0:
            continue
        
        chosen = cand_blocks.sort_values(["day", "shift", "room"]).iloc[0]
        need = int(prow["duration"]) + CLEANUP
        
        new_row = {
            "patient_id": int(pid),
            "room": int(chosen["room"]),
            "day": int(chosen["day"]),
            "shift": int(chosen["shift"]),
            "used_min": need,
            "surgeon_id": int(prow["surgeon_id"]),
            "iteration": -1,
            "W_patient": None,
            "W_block": None,
        }
        
        result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)
        ids_in_effective.append(int(pid))
    
    return result, ids_out, ids_in_effective, True


# ============================================================
#     MAIN ILS/VNS LOOP (5 minutos)
# ============================================================

print("\n" + "="*60)
print("     ILS/VNS — Iterated Local Search com Shaking")
print("="*60)

ILS_TIME_LIMIT = 2 * 60  # 15 minutos em segundos

# Ponto de partida: melhor solução do LS inicial
incumbent_assignments = best_assignments.copy()
incumbent_score = best_score
incumbent_feas = best_feas
incumbent_rooms = best_rooms_free.copy()

# Melhor global
global_best_assignments = incumbent_assignments.copy()
global_best_score = incumbent_score
global_best_feas = incumbent_feas
global_best_rooms = incumbent_rooms.copy()

print(f"ILS starting score: {incumbent_score:.4f}")
print(f"Time limit: {ILS_TIME_LIMIT // 60} minutes")
print(f"Max time per LS move: {MAX_TIME_PER_MOVE} seconds")

start_time = time.time()
# Continuar numeração das iterações do LS inicial
ils_iteration_counter = len(iteration_log) if iteration_log else 0
shaking_level = 1  # Começa no Shaking 1

# Contadores para estatísticas
shaking1_count = 0
shaking2_count = 0
improvements_count = 0
ils_iter = 0  # contador interno do ILS

while (time.time() - start_time) < ILS_TIME_LIMIT:
    ils_iter += 1
    ils_iteration_counter += 1
    elapsed = time.time() - start_time
    
    # ========== SHAKING (sempre no INCUMBENT - teoria VNS) ==========
    if shaking_level == 1:
        # Shaking 1: swap i-j com i=1-5, j=1-5
        shaking1_count += 1
        shaken_sol, ids_out, ids_in, success = shaking(
            incumbent_assignments, df_patients, df_rooms, df_surgeons, C_PER_SHIFT,
            min_out=3, max_out=5, min_in=3, max_in=5
        )
        shaking_name = "SHAKING_1"
    else:
        # Shaking 2: swap i-j com i=1-8, j=1-8 (mínimo 1)
        shaking2_count += 1
        shaken_sol, ids_out, ids_in, success = shaking(
            incumbent_assignments, df_patients, df_rooms, df_surgeons, C_PER_SHIFT,
            min_out=6, max_out=8, min_in=6, max_in=8
        )
        shaking_name = "SHAKING_2"
    
    # Verificar se o shaking foi bem-sucedido
    if not success:
        print(f"\n[ILS Iter {ils_iter}] {shaking_name} FALHOU: não há pacientes suficientes.")
        print(f"Aceitando a melhor solução encontrada até agora.")
        print(f"Best score: {global_best_score:.4f}")
        break
    
    # ========== LOCAL SEARCH (sempre executado) ==========
    # Calcular shaking score antes do LS
    feas_shaken = feasibility_metrics(shaken_sol, df_rooms, df_surgeons, df_patients, C_PER_SHIFT)
    rooms_shaken = feas_shaken["rooms_cap_join"].copy()
    rooms_shaken["utilization"] = rooms_shaken.apply(
        lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0, axis=1
    )
    assign_shaken = shaken_sol.merge(
        df_patients[["patient_id", "priority", "waiting", "duration"]],
        on="patient_id", how="left"
    )
    shaking_score = evaluate_schedule(assign_shaken, df_patients, rooms_shaken, feas_shaken["excess_block_min"])["score"]
    
    # Log do resultado do shaking (antes do LS)
    shaking_metrics = eval_components(assign_shaken, rooms_shaken, feas_shaken)
    shaking_id_str = f"S{ils_iter}"
    log_ils_iteration(
        fase=shaking_name,
        iteracao=shaking_id_str,
        metrics=shaking_metrics,
        ils_iteration=ils_iter,
        ls_phase="SHAKING",
        accepted=None,
        shaking_id=shaking_id_str
    )
    
    ls_result, ls_score, ls_feas, ls_rooms = run_local_search_phase(
        shaken_sol, df_patients, df_rooms, df_surgeons, C_PER_SHIFT,
        verbose=False,
        shaking_info={'name': shaking_name, 'iteration': ils_iter, 'removed': ids_out, 'added': ids_in},
        ils_iteration=ils_iter,
        shaking_id=shaking_id_str
    )
    
    # ========== ACCEPTANCE CRITERION ==========
    if ls_score > incumbent_score:
        # MELHORIA: aceita e atualiza incumbent
        improvements_count += 1
        
        # Print quando há melhoria
        print(f"\n[ILS Iter {ils_iter:4d}] {shaking_name}")
        print(f"  → Shaking score: {shaking_score:.4f} (removed={len(ids_out)}, added={len(ids_in)})")
        print(f"  → After LS: {ls_score:.4f} ✓ IMPROVED (+{ls_score-incumbent_score:.4f}) | elapsed={elapsed:.1f}s")
        
        # Atualiza incumbent
        incumbent_assignments = ls_result.copy()
        incumbent_score = ls_score
        incumbent_feas = ls_feas
        incumbent_rooms = ls_rooms.copy()
        
        # Atualizar melhor global
        if ls_score > global_best_score:
            global_best_score = ls_score
            global_best_assignments = ls_result.copy()
            global_best_feas = ls_feas
            global_best_rooms = ls_rooms.copy()
        
        # VNS: volta ao Shaking 1 quando melhora
        shaking_level = 1
        
    else:
        # SEM MELHORIA: avança na hierarquia VNS
        if shaking_level == 1:
            shaking_level = 2
        else:
            # Já estava no nível 2, volta ao 1
            shaking_level = 1
        
        # Log periódico (a cada 20 iterações)
        if ils_iter % 20 == 0:
            print(f"[ILS Iter {ils_iter:4d}] {shaking_name} → no improvement (score={ls_score:.4f}) | "
                  f"incumbent={incumbent_score:.4f} | next_level={shaking_level} | elapsed={elapsed:.1f}s")

# Fim do tempo
elapsed_total = time.time() - start_time
print("\n" + "="*60)
print("     ILS/VNS COMPLETED")
print("="*60)
print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.2f} min)")
print(f"Total ILS iterations: {ils_iter}")
print(f"Shaking 1 executions: {shaking1_count}")
print(f"Shaking 2 executions: {shaking2_count}")
print(f"Total improvements: {improvements_count}")
print(f"Improvement rate: {(improvements_count / ils_iter * 100):.1f}%")
print(f"Initial score (after LS): {best_score:.4f}")
print(f"Final ILS score: {global_best_score:.4f}")
print(f"Improvement: {((global_best_score - best_score) / best_score * 100):.2f}%")

# Atualizar best_assignments e best_score para o export final
best_assignments = global_best_assignments.copy()
best_score = global_best_score
# best_feas e best_rooms_free serão recalculados mais à frente a partir de best_assignments


# garantir que as chaves são int
best_assignments["patient_id"] = best_assignments["patient_id"].astype(int)
df_patients["patient_id"] = df_patients["patient_id"].astype(int)


final_print = best_assignments.merge(
    df_patients[["patient_id", "priority", "duration"]],
    on="patient_id",
    how="left"
).sort_values(["day", "shift", "room", "iteration"])

for _, row in final_print.iterrows():
    print(
        f"B{row['room']}_{row['day']}_{row['shift']} "
        f"(p={row['patient_id']}, "
        f"s={row['surgeon_id']}, "
        f"dur={row['duration']})"
    )
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

#print("\nSurgeons — remaining free minutes per day/shift:")
#print(df_surgeon_free.head(12))


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
    on="patient_id", how="left"
).sort_values("iteration")


# Add a simple sequence number per (room,day,shift)
assignments_enriched["seq_in_block"] = (
    assignments_enriched.groupby(["room", "day", "shift"]).cumcount() + 1
)



# ---------- 3) Capacity snapshots (final) ----------
# RECALCULAR rooms_free, surgeons_free e feasibility a partir do best_assignments final
# (não usar df_capacity que ficou desatualizado após ILS/VNS)

# Recalcular feasibility metrics a partir do best_assignments final
best_feas = feasibility_metrics(best_assignments, df_rooms, df_surgeons, df_patients, C_PER_SHIFT)

# Rooms: recalcular usado por bloco a partir de best_assignments
rooms_base = df_rooms[["room", "day", "shift", "available"]].copy()
rooms_base["cap_min"] = rooms_base["available"] * C_PER_SHIFT

if len(best_assignments):
    used_by_block = (best_assignments.groupby(["room", "day", "shift"], as_index=False)
                     .agg(used_min=("used_min", "sum")))
else:
    used_by_block = rooms_base[["room", "day", "shift"]].copy()
    used_by_block["used_min"] = 0

rooms_free = rooms_base.merge(used_by_block, on=["room", "day", "shift"], how="left").fillna({"used_min": 0})
rooms_free["free_min"] = (rooms_free["cap_min"] - rooms_free["used_min"]).clip(lower=0)
rooms_free["utilization"] = rooms_free.apply(
    lambda r: (r["used_min"] / r["cap_min"]) if r["cap_min"] > 0 else 0.0, axis=1
)
rooms_free = rooms_free.sort_values(["room", "day", "shift"]).reset_index(drop=True)

# Surgeons: recalcular carga por cirurgião a partir de best_assignments
surgeons_base = df_surgeons[["surgeon_id", "day", "shift", "available"]].drop_duplicates()

if len(best_assignments):
    surg_load_final = (best_assignments.groupby(["surgeon_id", "day", "shift"], as_index=False)
                       .agg(used_min=("used_min", "sum")))
else:
    surg_load_final = surgeons_base[["surgeon_id", "day", "shift"]].copy()
    surg_load_final["used_min"] = 0

surgeons_free = surgeons_base.merge(surg_load_final, on=["surgeon_id", "day", "shift"], how="left").fillna({"used_min": 0})
surgeons_free["cap_min"] = surgeons_free["available"] * C_PER_SHIFT
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
    
    # ======== ITERATIONS LOG - LS (LS1, LS2, LS3) ========
    if iteration_log:
        df_iter_log = pd.DataFrame(iteration_log).sort_values(["fase", "iteracao"])  # Agrupar por fase
    else:
        df_iter_log = pd.DataFrame([{"info": "Sem iterações LS registadas"}])
    df_iter_log.to_excel(writer, sheet_name="Iterations_Log", index=False)
    
    # ======== ITERATIONS LOG - ILS/VNS ========
    if ils_log:
        df_ils_log = pd.DataFrame(ils_log)
        # Ordenar por ils_iteration (se existir) e depois por ls_phase para manter a sequência
        if 'ils_iteration' in df_ils_log.columns:
            df_ils_log = df_ils_log.sort_values(["ils_iteration"], kind='stable')
    else:
        df_ils_log = pd.DataFrame([{"info": "Sem iterações ILS registadas"}])
    df_ils_log.to_excel(writer, sheet_name="IterationsILS_Log", index=False)

print(f"\nExcel exported → {xlsx_path}")
print(f"Initial score = {score_init['score']:.4f}, feasibility_score = {feas_init['feasibility_score']}")
