# -*- coding: utf-8 -*-
"""
Operating Room Scheduling — Step 1 + Step 2 (Iterative Dispatching Rule)
Scenario 2: Surgeon can change room within the same shift
Author: Joana
"""

from pathlib import Path
import re
import ast
import itertools
import pandas as pd
import random
import numpy as np

# ================== GLOBAL ITERATION LOG ==================
global_iter_log = []
global_iter_counter = 0

# Fix seeds
random.seed(42)
np.random.seed(42)
# ------------------------------
# PARAMETERS
# ------------------------------
DATA_FILE = "Instance_C1_30.dat"

C_PER_SHIFT = 360   # minutes per shift (6h * 60)
CLEANUP = 17        # cleaning time 

ALPHA1 = 0.25 # priority
ALPHA2 = 0.25  # waited days
ALPHA3 = 0.25 # deadline closeness
ALPHA4 = 0.25 # feasible blocks

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
   

    # capacidade por bloco (inclui free_min)
    cap_ok = df_capacity[df_capacity["available"]==1].copy()

    # ADICIONAR cap_min (porque df_capacity não tem esta coluna)
    cap_ok["cap_min"] = C_PER_SHIFT

    # need = duração + cleanup + mudança de sala se aplicável
    cap_ok["need"] = need

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

def normalize_and_sequence(assignments,
                           df_patients,
                           C_PER_SHIFT,
                           CLEANUP,
                           TOLERANCE,
                           ROOM_CHANGE_TIME):
    """
    Aplica SEM ALTERAR A LÓGICA:
      - remove colunas clínicas duplicadas
      - faz merge com df_patients
      - aplica sequence_global_by_surgeon
      - devolve apenas scheduled_by_seq == 1
    """

    # 1) remover colunas clínicas duplicadas
    drop_cols = [c for c in ["duration", "priority", "waiting"] if c in assignments.columns]
    base = assignments.drop(columns=drop_cols).copy()

    # 2) merge com pacientes
    enriched = base.merge(
        df_patients[["patient_id", "duration", "priority", "waiting"]],
        on="patient_id",
        how="left"
    )

    # 3) sequenciamento global
    enriched = sequence_global_by_surgeon(
        enriched,
        C_PER_SHIFT=C_PER_SHIFT,
        CLEANUP=CLEANUP,
        TOLERANCE=TOLERANCE,
        ROOM_CHANGE_TIME=ROOM_CHANGE_TIME
    )

    # 4) manter apenas os casos efetivamente sequenciados
    seq = enriched[enriched["scheduled_by_seq"] == 1].copy()

    return seq

def normalize_and_resequence(assignments_with_order,
                             df_patients,
                             C_PER_SHIFT,
                             CLEANUP,
                             TOLERANCE,
                             ROOM_CHANGE_TIME):
    """
    Usada APENAS no LS4.
    Assume que order_hint JÁ EXISTE e é o que define a ordem.
    Só recalcula tempos (start/end).
    """

    # remover colunas clínicas duplicadas (mas NÃO order_hint)
    drop_cols = [c for c in ["duration", "priority", "waiting"] if c in assignments_with_order.columns]
    base = assignments_with_order.drop(columns=drop_cols).copy()

    # merge clínico
    enriched = base.merge(
        df_patients[["patient_id", "duration", "priority", "waiting"]],
        on="patient_id",
        how="left"
    )

    # sequenciador usa order_hint
    enriched = sequence_global_by_surgeon(
        enriched,
        C_PER_SHIFT=C_PER_SHIFT,
        CLEANUP=CLEANUP,
        TOLERANCE=TOLERANCE,
        ROOM_CHANGE_TIME=ROOM_CHANGE_TIME
    )

    return enriched[enriched["scheduled_by_seq"] == 1].copy()

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
            base_wait_term = (float(merged["waiting"].mean()) / wmax) if wmax > 0 else 0.0
        else:
            base_wait_term = 0.0

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


import json  # <-- ADICIONAR AOS IMPORTS

# ================== SENSITIVITY LOGGING (ANÁLISE DE SENSIBILIDADE) ==================
EVAL_WEIGHTS = (0.6, 0.1, 0.25, 0.05)  # manter em linha com evaluate_schedule
improvement_log = []
iteration_log = []

def log_iteration(fase, iteracao, metrics, accepted):
    """
    Regista apenas as colunas pedidas pelo utilizador.
    """
    iteration_log.append({
        "fase": fase,
        "iteracao": int(iteracao),
        "score": float(metrics["score"]),
        "assigned_patients": int(metrics["assigned_patients"]),
        "ratio_scheduled_raw": float(metrics["ratio_scheduled_raw"]),
        "util_rooms_raw": float(metrics["util_rooms_raw"]),
        "avg_waiting_raw": float(metrics["avg_waiting_raw"]),
        "avg_priority_raw": float(metrics["avg_priority_raw"]),
        "deadline_overdue_patients": int(metrics["deadline_overdue_patients"]),
        "excess_block_min_raw": int(metrics["excess_block_min_raw"]),
        "excess_surgeon_min_raw": int(metrics["excess_surgeon_min_raw"]),
        "accepted": int(bool(accepted)),  # Última coluna: 1 se aceite, 0 se não
    })

def log_global_iteration(fase, iteracao_local, assignments_seq, rooms_free, feas):
    global global_iter_counter
    global_iter_counter += 1

    metrics = eval_components(assignments_seq, rooms_free, feas)

    global_iter_log.append({
        "fase": fase,
        "iteracao": global_iter_counter,   # ordem GLOBAL
        "score": metrics["score"],

        # ===== KPIs pedidos =====
        "n_patient": metrics["assigned_patients"],

        # r — utilização média das salas
        "scheduled_rooms_r": metrics["util_rooms_raw"],

        # k — waiting médio
        "u_waiting_k": metrics["avg_waiting_raw"],

        # j — prioridade média
        "p_priority_j": metrics["avg_priority_raw"],

        # l — nº pacientes overdue
        "w_overdue_l": metrics["deadline_overdue_patients"],

        # m — excesso bloco
        "f_block_m": metrics["excess_block_min_raw"],

        # excesso cirurgião
        "surgeon_min_raw": metrics["excess_surgeon_min_raw"],
    })


def eval_components(assignments_seq, rooms_free, feas, weights=EVAL_WEIGHTS):
    """
    Calcula métricas detalhadas para exportar no iteration_log
    e também as componentes do score necessárias para o improvement_log.
    """

    total_patients = len(df_patients)

    # -------- Pacientes agendados --------
    assigned_patients = int(assignments_seq["patient_id"].nunique()) if len(assignments_seq) else 0
    ratio_scheduled_raw = assigned_patients / total_patients if total_patients > 0 else 0.0

    # -------- Utilização real --------
    util_rooms_raw = float(
        rooms_free.loc[rooms_free["cap_min"] > 0, "utilization"].mean()
    ) if len(rooms_free) else 0.0

    # -------- Priority & Waiting médios --------
    if len(assignments_seq):
        missing_cols = [c for c in ["priority", "waiting"] if c not in assignments_seq.columns]
        if missing_cols:
            merged = assignments_seq.merge(
                df_patients[["patient_id", "priority", "waiting"]],
                on="patient_id", how="left"
            )
        else:
            merged = assignments_seq

        avg_priority_raw = float(merged["priority"].mean())
        avg_waiting_raw = float(merged["waiting"].mean())
    else:
        avg_priority_raw = 0.0
        avg_waiting_raw = 0.0

    # -------- Overdues APENAS entre os pacientes agendados --------
    # -------- Overdues APENAS entre pacientes AGENDADOS --------
    if len(assignments_seq):
    
        # Se priority/waiting não existem (sequenciador removeu-as), fazemos merge
        cols_needed = ["priority", "waiting"]
        missing_cols = [c for c in cols_needed if c not in assignments_seq.columns]
    
        if missing_cols:
            pats_assigned = assignments_seq.merge(
                df_patients[["patient_id", "priority", "waiting"]],
                on="patient_id",
                how="left"
            )
        else:
            pats_assigned = assignments_seq.copy()
    
        # Agora priority e waiting EXISTEM de certeza
        pats_assigned["deadline_limit"] = pats_assigned["priority"].apply(deadline_limit_from_priority)
        pats_assigned["overdue_days"] = (pats_assigned["waiting"] - pats_assigned["deadline_limit"]).clip(lower=0)
        deadline_overdue_patients = int((pats_assigned["overdue_days"] > 0).sum())
    
    else:
        deadline_overdue_patients = 0


    # -------- Excessos --------
    excess_block_min_raw = int(feas["excess_block_min"])
    excess_surgeon_min_raw = int(feas["excess_surgeon_min"])

    # -------- Componentes do score --------
    ev = evaluate_schedule(
        assignments=assignments_seq,
        patients=df_patients,
        rooms_free=rooms_free,
        excess_block_min=feas["excess_block_min"],
        weights=weights
    )

    return {
        # === termos do score ===
        "score": float(ev["score"]),
        "ratio_scheduled": float(ev["ratio_scheduled"]),
        "util_rooms": float(ev["util_rooms"]),
        "prio_rate": float(ev["prio_rate"]),
        "norm_wait_term": float(ev["norm_wait_term"]),
        "excess_block_min": excess_block_min_raw,

        # === métricas “raw” ===
        "assigned_patients": assigned_patients,
        "ratio_scheduled_raw": ratio_scheduled_raw,
        "util_rooms_raw": util_rooms_raw,
        "avg_waiting_raw": avg_waiting_raw,
        "avg_priority_raw": avg_priority_raw,
        "deadline_overdue_patients": deadline_overdue_patients,
        "excess_block_min_raw": excess_block_min_raw,
        "excess_surgeon_min_raw": excess_surgeon_min_raw,
    }





   

def summarize_change(prev, new, weights=EVAL_WEIGHTS):
    """Computa deltas e identifica o principal driver da mudança do score."""
    w1, w2, w3, w4 = weights
    terms = {
        "ratio_scheduled":        w1 * (new["ratio_scheduled"] - prev["ratio_scheduled"]),
        "util_rooms":             w2 * (new["util_rooms"] - prev["util_rooms"]),
        "prio_rate":              w3 * (new["prio_rate"] - prev["prio_rate"]),
        "norm_wait_term":         w4 * (new["norm_wait_term"] - prev["norm_wait_term"]),
        "excess_block_penalty":  -0.001 * (new["excess_block_min"] - prev["excess_block_min"]),
    }
    main_driver = max(terms, key=lambda k: abs(terms[k]))
    return {
        "Δ_score":            float(new["score"] - prev["score"]),
        "Δ_ratio_scheduled":  float(new["ratio_scheduled"] - prev["ratio_scheduled"]),
        "Δ_util_rooms":       float(new["util_rooms"] - prev["util_rooms"]),
        "Δ_prio_rate":        float(new["prio_rate"] - prev["prio_rate"]),
        "Δ_norm_wait_term":   float(new["norm_wait_term"] - prev["norm_wait_term"]),
        "Δ_excess_block_min": int(new["excess_block_min"] - prev["excess_block_min"]),
        "main_driver":        main_driver,
        "main_driver_contrib":float(terms[main_driver]),
    }

def _to_json_str(obj):
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(obj)

def log_improvement(fase, iteracao, prev_metrics, new_metrics,
                    acao_curta, acao_detalhe):
    """Regista uma linha no log de melhorias de score."""
    deltas = summarize_change(prev_metrics, new_metrics)
    improvement_log.append({
        "fase": fase,
        "iteracao": int(iteracao),
        "score_prev": float(prev_metrics["score"]),
        "score_novo": float(new_metrics["score"]),
        "delta_score": float(new_metrics["score"] - prev_metrics["score"]),
        "acao": acao_curta,
        "acao_detalhe_json": _to_json_str(acao_detalhe),

        # estado anterior (para comparação rápida no Excel)
        "prev_assigned": prev_metrics["assigned_patients"],
        "prev_ratio_scheduled": prev_metrics["ratio_scheduled"],
        "prev_util_rooms": prev_metrics["util_rooms"],
        "prev_prio_rate": prev_metrics["prio_rate"],
        "prev_norm_wait_term": prev_metrics["norm_wait_term"],
        "prev_excess_block_min": prev_metrics["excess_block_min"],

        # estado novo
        "new_assigned": new_metrics["assigned_patients"],
        "new_ratio_scheduled": new_metrics["ratio_scheduled"],
        "new_util_rooms": new_metrics["util_rooms"],
        "new_prio_rate": new_metrics["prio_rate"],
        "new_norm_wait_term": new_metrics["norm_wait_term"],
        "new_excess_block_min": new_metrics["excess_block_min"],

        # deltas + driver principal
        **deltas,
    })
# ================== /SENSITIVITY LOGGING ==================


def generate_neighbor_swap(current_assignments,         #Swap com remoção/admissão de pacientes (swap i↔j no ILS)
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
     
        O algoritmo copia o agendamento atual, escolhe aleatoriamente até max_swap_out pacientes
        já agendados para remover, embaralha a lista de não agendados e tenta inserir até max_swap_in deles. 
        Para cada novo paciente candidato, calcula blocos viáveis com candidate_blocks_for_patient_in_solution, 
        sorteia um bloco, cria um novo registo e concatena ao dataframe, registrando quem saiu (ids_out) e quem entrou de facto (ids_in_effective).
      
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

        chosen = cand_blocks.sample(1).iloc[0]
        need = int(prow["duration"]) + CLEANUP

        new_row = {
            "patient_id": int(pid),
            "room": int(chosen["room"]),
            "day": int(chosen["day"]),
            "shift": int(chosen["shift"]),
            "used_min": need,
            "surgeon_id": int(prow["surgeon_id"]),
            "iteration": -1,
            "W_patient": 0.0,
            "W_block": 0.0,
        }

        assignments = pd.concat(
            [assignments, pd.DataFrame([new_row])],
            ignore_index=True
        )

        ids_in_effective.append(int(pid))

    return assignments, ids_in_effective



"""
Cálculo dos blocos viáveis para inserir um paciente
A função candidate_blocks_for_patient_in_solution (ver a seguir) verifica disponibilidade do cirurgião e das salas,
subtrai a carga já usada para obter free_min, e filtra apenas blocos onde o tempo necessário (duração + limpeza) cabe tanto na sala quanto no limite do cirurgião. 
O resultado é a lista de (room, day, shift) possíveis usada pelo swap anterior
"""

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

    # >>> UNIFORMIZAR: aceitar tolerância no BLOCO
    # (equivalente a used_min + need <= C_PER_SHIFT + TOLERANCE)
    cap_ok = rooms_join[
        (rooms_join["available"] == 1) &
        ((rooms_join["used_min"] + need) <= C_PER_SHIFT + TOLERANCE)
    ][["room","day","shift","free_min","used_min"]].rename(columns={"used_min": "block_used"})

    cand = surg_ok.merge(cap_ok, on=["day","shift"], how="inner")

    # carga do cirurgião por (day,shift)
    if len(assignments):
        sload = (assignments[assignments["surgeon_id"] == sid]
                 .groupby(["day","shift"], as_index=False)
                 .agg(used_min=("used_min","sum")))
    else:
        sload = pd.DataFrame(columns=["day","shift","used_min"])

    sload = sload.rename(columns={"used_min": "surg_used"})
    cand = cand.merge(sload, on=["day","shift"], how="left").fillna({"surg_used": 0})

    # >>> UNIFORMIZAR: aceitar tolerância no CIRURGIÃO
    cand = cand[(cand["surg_used"] + need) <= C_PER_SHIFT + TOLERANCE]

    # (opcional, redundante, mas deixa explícito)
    cand = cand[(cand["block_used"] + need) <= C_PER_SHIFT + TOLERANCE]

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

    #NOVO: verificar se existe coluna order_hint
    has_order_hint = "order_hint" in df.columns

    for (day, shift), df_ds in df.groupby(["day", "shift"], sort=False):

        # construir filas por sala
        room_queues = {}
        for room, df_room in df_ds.groupby("room", sort=False):
            df_room = df_room.copy()

            if has_order_hint and df_room["order_hint"].notna().any():
                # se existir order_hint, usamos essa ordem
                df_room_ordered = df_room.sort_values("order_hint", kind="mergesort")
            else:
                #caso contrário usa a heurística antiga (por cirurgião + duração)
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
                else:
                    df_room_ordered = df_room.copy()

            room_queues[room] = list(df_room_ordered["orig_idx"].values)

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

                room_ready = t_room[room]
                base_surg_ready = t_surg[sid]

                # penalização de mudança de sala
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

def full_evaluation(assignments_seq):
    """
    Avalia uma solução JÁ SEQUENCIADA.
    Assume que:
      - start_min / end_min existem
      - scheduled_by_seq == 1
    """

    # garantir dados clínicos
    missing = [c for c in ["duration", "priority", "waiting"] if c not in assignments_seq.columns]
    if missing:
        assignments_seq = assignments_seq.merge(
            df_patients[["patient_id", "duration", "priority", "waiting"]],
            on="patient_id",
            how="left"
        )

    # capacidade das salas
    rooms_free = build_room_free_from_assignments(
        assignments=assignments_seq,
        df_rooms=df_rooms,
        C_PER_SHIFT=C_PER_SHIFT
    )

    # viabilidade
    feas = feasibility_metrics(
        assignments=assignments_seq,
        df_rooms=df_rooms,
        df_surgeons=df_surgeons,
        patients=df_patients,
        C_PER_SHIFT=C_PER_SHIFT
    )

    # score
    ev = evaluate_schedule(
        assignments=assignments_seq,
        patients=df_patients,
        rooms_free=rooms_free,
        excess_block_min=feas["excess_block_min"]
    )

    return ev["score"], rooms_free, feas


"""
No generate_neighbor_resequence (em baixo), o movimento não troca pacientes de bloco; 
apenas altera a coluna order_hint para alguns blocos com pelo menos dois casos. 
Ele garante uma ordem base, escolhe blocos aleatórios, e dentro de cada um faz swaps de order_hint entre pares aleatórios de cirurgias para mudar a sequência de execução, retornando também um log do “antes/depois
"""

def generate_neighbor_resequence(current_enriched,
                                 max_blocks_to_change=1,
                                 swaps_per_block=1):
    """
    Gera um vizinho mexendo apenas na ordem (order_hint) dentro de alguns blocos.
    Não altera que paciente está em que room/day/shift.

    Devolve:
      - neigh: novo DataFrame com order_hint alterado
      - change_log: lista com info das mudanças por bloco
    """
    neigh = current_enriched.copy()

    # se ainda não existir, criamos uma ordem base determinística por bloco
    if "order_hint" not in neigh.columns:
        neigh["order_hint"] = (
            neigh.groupby(["room", "day", "shift"])
                 .cumcount()
        )

    # escolher blocos com pelo menos 2 casos
    blocks = (
        neigh.groupby(["room", "day", "shift"])
             .filter(lambda g: len(g) >= 2)
             .groupby(["room", "day", "shift"])
    )
    block_keys = list(blocks.groups.keys())
    if not block_keys:
        return neigh, []  # nada a fazer

    random.shuffle(block_keys)
    block_keys = block_keys[:max_blocks_to_change]

    change_log = []

    for key in block_keys:
        g_idx = blocks.groups[key]

        # ordem antes (por order_hint)
        before_order = (
            neigh.loc[g_idx]
                 .sort_values("order_hint")
                 [["patient_id", "surgeon_id", "order_hint"]]
                 .to_dict(orient="records")
        )

        # fazer alguns swaps de order_hint
        for _ in range(swaps_per_block):
            if len(g_idx) < 2:
                break
            i, j = random.sample(list(g_idx), 2)

            oi, oj = neigh.at[i, "order_hint"], neigh.at[j, "order_hint"]
            neigh.at[i, "order_hint"], neigh.at[j, "order_hint"] = oj, oi

        # ordem depois
        after_order = (
            neigh.loc[g_idx]
                 .sort_values("order_hint")
                 [["patient_id", "surgeon_id", "order_hint"]]
                 .to_dict(orient="records")
        )

        change_log.append({
            "room": int(key[0]),
            "day": int(key[1]),
            "shift": int(key[2]),
            "before": before_order,
            "after": after_order,
        })

    return neigh, change_log

"""
A função em baixo seleciona dois casos distintos que estejam no mesmo dia e turno, mas em salas diferentes. 
Depois simplesmente troca o número da sala entre eles, preservando dia e turno, e devolve o vizinho com informações do swap aplicado. 
É um movimento 1–por–1, pensado para manter quase sempre a viabilidade temporal dos blocos
"""

def generate_neighbor_cross_room_swap(current_assignments):
    """
    CROSS-ROOM SWAP (até 2x2):
  troca blocos de 1 ou 2 cirurgias entre duas salas do MESMO dia e turno.
  Possibilidades: 1↔1, 2↔1, 1↔2 ou 2↔2.
    
    Retorna:
      - neighbor: novo dataframe com swap aplicado
      - swap_info: dicionário com info do swap (para imprimir)
    """
    a = current_assignments.copy()

    if len(a) < 2:
        return a, None

    # candidatos: pares de salas no mesmo dia/turno com cirurgias
    sched = a.copy()
    
    block_pairs = []
    for (day, shift), df_ds in sched.groupby(["day", "shift"]):
        rooms = df_ds["room"].unique().tolist()
        if len(rooms) < 2:
            continue
        for roomA, roomB in itertools.combinations(rooms, 2):
            df_a = df_ds[df_ds["room"] == roomA]
            df_b = df_ds[df_ds["room"] == roomB]
            if df_a.empty or df_b.empty:
                continue
            block_pairs.append((day, shift, roomA, roomB, df_a, df_b))

    # escolher pares que estejam no MESMO dia e shift mas SALAS diferentes
    if not block_pairs:
        return a, None

    # escolher um par aleatório
    day, shift, roomA, roomB, df_a, df_b = random.choice(block_pairs)

    size_options = [(1, 1), (2, 1), (1, 2), (2, 2)]
    size_options = [
        (sa, sb) for sa, sb in size_options
        if len(df_a) >= sa and len(df_b) >= sb
    ]
    if not size_options:
        return a, None

    # swap das salas
    size_a, size_b = random.choice(size_options)
    patients_a = random.sample(df_a["patient_id"].tolist(), size_a)
    patients_b = random.sample(df_b["patient_id"].tolist(), size_b)

    neighbor = a.copy()
    neighbor.loc[neighbor["patient_id"].isin(patients_a), "room"] = roomB
    neighbor.loc[neighbor["patient_id"].isin(patients_b), "room"] = roomA

    # validar capacidade após o swap (cada bloco não pode ultrapassar C_PER_SHIFT + TOLERANCE)
    def block_capacity(room):
        row = df_rooms[(df_rooms["room"] == room) & (df_rooms["day"] == day) & (df_rooms["shift"] == shift)]
        if row.empty:
            return 0
        return C_PER_SHIFT if int(row.iloc[0]["available"]) == 1 else 0

    cap_ok = True
    for room in (roomA, roomB):
        used = neighbor[(neighbor["room"] == room) & (neighbor["day"] == day) & (neighbor["shift"] == shift)]["used_min"].sum()
        if used > block_capacity(room) + TOLERANCE:
            cap_ok = False
            break

    if not cap_ok:
        return a, None

    swap_info = {
        "day": int(day),
        "shift": int(shift),
        "roomA": int(roomA),
        "roomB": int(roomB),
        "from_roomA_to_roomB": [int(pid) for pid in patients_a],
        "from_roomB_to_roomA": [int(pid) for pid in patients_b],       
    }

    return neighbor, swap_info

def shaking(solution,
            k,
            df_patients,
            df_rooms,
            df_surgeons,
            C_PER_SHIFT):
    """
    k = 1 → swap 3–5
    k = 2 → swap 6–8
    """

    if k == 1:
        max_out = random.randint(3, 5)
        max_in  = random.randint(3, 5)
    else:
        max_out = random.randint(6, 8)
        max_in  = random.randint(6, 8)

    shaken, _, _ = generate_neighbor_swap(
        solution,
        df_patients,
        df_rooms,
        df_surgeons,
        C_PER_SHIFT,
        max_swap_out=max_out,
        max_swap_in=max_in
    )

    print(f"  SHAKING {k}: swap_out={max_out}, swap_in={max_in}")

    return shaken

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

    df_p_blocks = df_p_blocks.copy()
    df_p_blocks["overflow_min"] = (
        (df_p_blocks["used_min"] + df_p_blocks["need"]) - C_PER_SHIFT
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
    
# ---------- GUARDAR SOLUÇÃO INICIAL (ANTES DO LS) ----------
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

log_global_iteration(
    "INITIAL_HEURISTIC",
    0,
    assignments_seq_view,
    initial_rooms_free,
    initial_feas
)
   
# =========================================================
#       FIRST LOCAL SEARCH - swap i - j 
# =========================================================

N_ILS_ITER = 100
MAX_SWAP_OUT = 2
MAX_SWAP_IN  = 2

# ponto de partida: solução JÁ sequenciada da heurística
current_assignments = assignments_seq_view.copy()

# avaliação inicial (já está sequenciada)
current_score, current_rooms_free, current_feas = \
    full_evaluation(current_assignments)

best_score = current_score
best_assignments = current_assignments.copy()
best_rooms_free = current_rooms_free.copy()
best_feas = current_feas  

print("\nILS START")
print("Initial score:", current_score)

# Limpar iteration_log para conter apenas Local Search e inicializar contador
iteration_log = []
ils_iteration_counter = 0

for it in range(N_ILS_ITER):
    ils_iteration_counter += 1

    # -------- estado ANTES --------
    prev_score = current_score
    prev_metrics = eval_components(
        current_assignments, current_rooms_free, current_feas
    )

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

    # -------- 2) SEQUENCING EXPLÍCITO --------
    neighbor_seq = normalize_and_sequence(
        neighbor_struct,
        df_patients,
        C_PER_SHIFT,
        CLEANUP,
        TOLERANCE,
        ROOM_CHANGE_TIME
    ).sort_values(
        ["day", "shift", "room", "seq_in_block"]
    ).reset_index(drop=True)

    # -------- 3) AVALIAR --------
    neigh_score, neigh_rooms_free, neigh_feas = \
        full_evaluation(neighbor_seq)

    new_metrics = eval_components(
        neighbor_seq, neigh_rooms_free, neigh_feas
    )

    accepted = (neigh_score > prev_score)

    # ✅ LOG SEMPRE
    log_iteration(
        "ILS1_SWAP",
        ils_iteration_counter,
        new_metrics,
        accepted=accepted
    )

    # -------- 4) ACEITAR SE MELHORA --------
    if accepted:
        current_assignments = neighbor_seq.copy()
        current_score = neigh_score
        current_rooms_free = neigh_rooms_free
        current_feas = neigh_feas

        # snapshot global
        log_global_iteration(
            "LS1_SWAP",
            it,
            current_assignments,
            current_rooms_free,
            current_feas
        )

        # melhor global do LS1
        if neigh_score > best_score:
            best_score = neigh_score
            best_assignments = neighbor_seq.copy()
            best_rooms_free = neigh_rooms_free.copy()
            best_feas = neigh_feas


# ============================================================
#     LS #2 — ADD-ONLY
# ============================================================

print("\n\n========== STARTING ILS #2 — ADD-ONLY ==========\n")

current_assignments = best_assignments.copy()
current_score, current_rooms_free, current_feas = \
    full_evaluation(current_assignments)

best_seq = current_assignments.copy()  # because current_assignments is already sequenced

best_score = current_score
best_assignments = current_assignments.copy()
best_rooms_free = current_rooms_free.copy()
best_feas = current_feas

N_ILS2_ITER = 100

print("Initial add-only score:", current_score)

# Resetar contador para LS2
ils_iteration_counter = 0

for it in range(N_ILS2_ITER):
    ils_iteration_counter += 1

    # estado ANTES
    _sc, _rooms, _feas = full_evaluation(current_assignments)
    prev_metrics = eval_components(current_assignments, _rooms, _feas)


    neighbor, ids_added = generate_neighbor_add_only(
        current_assignments, df_patients, df_rooms, df_surgeons, C_PER_SHIFT, max_add=2
    )
    
    # Se não conseguiu adicionar nada, logar como rejeitado e continuar
    if not ids_added:
        log_iteration("ILS2_ADD_ONLY", ils_iteration_counter, prev_metrics, accepted=False)
        continue

    # ---- sequenciar o neighbor ANTES de avaliar ----
    neighbor_seq = normalize_and_sequence(
        neighbor,
        df_patients,
        C_PER_SHIFT,
        CLEANUP,
        TOLERANCE,
        ROOM_CHANGE_TIME
    ).sort_values(
        ["day","shift","room","seq_in_block"]
    ).reset_index(drop=True)

    
    # ---- avaliar (3 outputs) ----
    neigh_score, neigh_rooms_free, neigh_feas = full_evaluation(neighbor_seq)
    new_metrics = eval_components(neighbor_seq, neigh_rooms_free, neigh_feas)

    log_iteration("ILS2_ADD_ONLY", ils_iteration_counter, new_metrics, accepted=(neigh_score > current_score))

    if neigh_score > current_score:
        current_assignments = neighbor_seq.copy()
        current_score = neigh_score
        current_rooms_free = neigh_rooms_free
        current_feas = neigh_feas
        log_global_iteration("LS2_ADD_ONLY", it, current_assignments, current_rooms_free, current_feas)

        # LOG
        acao_curta = f"add-only: +{len(ids_added)}"
        acao_detalhe = {"adicionados": ids_added}
        log_improvement("ILS2_ADD_ONLY", it, prev_metrics, new_metrics, acao_curta, acao_detalhe)

        if neigh_score > best_score:
            best_score = neigh_score
            best_assignments = neighbor.copy()
            best_seq = neighbor_seq.copy()
            best_rooms_free = neigh_rooms_free.copy()
            best_feas = neigh_feas

        print(f"[ILS2 Iter {it}] Improved score to {current_score:.4f} | added={ids_added}")
   
# ============================================================
#     CONSTRUIR A SOLUÇÃO FINAL (COM SEQUENCIAMENTO)
# ============================================================

# 1) Merge com pacientes — remover possíveis duplicadas para evitar duration_x/duration_y
drop_cols = [c for c in ["duration", "priority", "waiting"] if c in best_assignments.columns]
best_assignments_base = best_assignments.drop(columns=drop_cols)

best_assignments_enriched = best_assignments_base.merge(
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


# =========================================================
#        LS #3 — CROSS-ROOM SWAP (change surgeries from one room to another within the same shift )
# =========================================================

print("\n\n========== STARTING LS #3 — CROSS-ROOM SWAP ==========\n")

# ponto de partida = soluçao depois de sequenciar
current_assignments =assignments_seq_view.copy()

# avaliar solução inicial para esta LS
current_score, current_rooms_free, current_feas = \
    full_evaluation(current_assignments)


best_score = current_score
best_assignments_4 = current_assignments.copy()
best_seq_4 = current_assignments.copy()
best_rooms_free_4 = current_rooms_free.copy()
best_feas_4 = current_feas


print("Initial score (LS3):", current_score)

N_ILS4_ITER = 100

# Resetar contador para LS3
ils_iteration_counter = 0

for it in range(N_ILS4_ITER):
    ils_iteration_counter += 1

    # estado ANTES
    prev_score, prev_rooms, prev_feas = full_evaluation(current_assignments)
    prev_metrics = eval_components(current_assignments, prev_rooms, prev_feas)


    neighbor, swap_info = generate_neighbor_cross_room_swap(current_assignments)
    
    # Se não conseguiu fazer swap, logar como rejeitado e continuar
    if swap_info is None:
        log_iteration("LS3_CROSS_ROOM", ils_iteration_counter, prev_metrics, accepted=False)
        continue

    neighbor_seq = normalize_and_sequence(
        neighbor,
        df_patients,
        C_PER_SHIFT,
        CLEANUP,
        TOLERANCE,
        ROOM_CHANGE_TIME
    )

    
    neigh_score, neigh_rooms_free, neigh_feas = full_evaluation(neighbor_seq)
    new_metrics = eval_components(neighbor_seq, neigh_rooms_free, neigh_feas)

    log_iteration("LS3_CROSS_ROOM", ils_iteration_counter, new_metrics, accepted=(neigh_score > current_score))

    if neigh_score > current_score:
        current_assignments = neighbor_seq.copy()
        current_score = neigh_score
        current_rooms_free = neigh_rooms_free
        current_feas = neigh_feas
        log_global_iteration("LS3_CHANGE_ROOM", it, current_assignments, current_rooms_free, current_feas)
        # LOG
        acao_curta = (f"cross-room swap: day={swap_info['day']} shift={swap_info['shift']} "
                      f"{swap_info['roomA']}↔{swap_info['roomB']}")
        acao_detalhe = swap_info
        log_improvement("LS3_CROSS_ROOM", it, prev_metrics, new_metrics, acao_curta, acao_detalhe)

        improved_global = False
        if neigh_score > best_score:
            improved_global = True
            best_score = neigh_score
            best_assignments_4 = neighbor.copy()
            best_seq_4 = neighbor_seq.copy()
            best_rooms_free_4 = neigh_rooms_free.copy()
            best_feas_4 = neigh_feas

        print(f"[ILS3 Iter {it}] {'GLOBAL' if improved_global else 'LOCAL'} score improved: {neigh_score:.4f}")
        print(f"   ↪ swap day={swap_info['day']}, shift={swap_info['shift']} | "
              f"room {swap_info['roomA']} -> {swap_info['roomB']} (p {swap_info['from_roomA_to_roomB']}) | "
              f"room {swap_info['roomB']} -> {swap_info['roomA']} (p {swap_info['from_roomB_to_roomA']})")

print("\n========== END OF ILS #3 ==========\n")


assignments_seq_view = best_seq_4.copy()
df_room_free = best_rooms_free_4.copy()
best_feas = best_feas_4


# ============================================================
#     CONSTRUIR A SOLUÇÃO FINAL (COM SEQUENCIAMENTO)
# ============================================================

# 1) remover colunas duplicadas antes do merge
drop_cols = [c for c in ["duration", "priority", "waiting"] if c in assignments_seq_view.columns]
base_ls3 = assignments_seq_view.drop(columns=drop_cols)

best_assignments_enriched = base_ls3.merge(
    df_patients[["patient_id","duration","priority","waiting"]],
    on="patient_id",
    how="left"
)

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



# =========================================================
#        LS #4 — RESEQUENCING (mudar a ordem dentro dos blocos)
# =========================================================
print("\n\n========== STARTING LS #4 — RESEQUENCE ==========\n")

# 0) Construir current_enriched a partir da melhor solução até aqui
current_enriched = assignments_seq_view.copy()

# Garantir que temos colunas clínicas (se faltarem, faz merge)
if not {'duration', 'priority', 'waiting'}.issubset(current_enriched.columns):
    current_enriched = current_enriched.merge(
        df_patients[['patient_id', 'duration', 'priority', 'waiting']],
        on='patient_id',
        how='left'
    )

# Garantir que temos colunas de sequenciamento (se faltarem, sequencia)
if not {'start_min', 'end_min', 'seq_in_block', 'scheduled_by_seq'}.issubset(current_enriched.columns):
    current_enriched = sequence_global_by_surgeon(
        current_enriched,
        C_PER_SHIFT=C_PER_SHIFT,
        CLEANUP=CLEANUP,
        TOLERANCE=TOLERANCE,
        ROOM_CHANGE_TIME=ROOM_CHANGE_TIME
    )

# Inicializar uma ordem base: usar a sequência atual como pista
current_enriched['order_hint'] = current_enriched['seq_in_block']

# 1) Criar assignments "limpo" a partir do enriched (MANTER order_hint!)
drop_ls4_cols = [
    "duration", "priority", "waiting",
    "start_min", "end_min", "seq_in_block", "scheduled_by_seq"
]
current_assignments_ls4 = current_enriched.drop(
    columns=[c for c in drop_ls4_cols if c in current_enriched.columns]
).copy()


# 2) SEQUENCIAR a solução de partida do LS4 (OBRIGATÓRIO)
tmp = current_assignments_ls4.merge(
    df_patients[["patient_id","duration","priority","waiting"]],
    on="patient_id",
    how="left"
)

tmp = sequence_global_by_surgeon(
    tmp,
    C_PER_SHIFT=C_PER_SHIFT,
    CLEANUP=CLEANUP,
    TOLERANCE=TOLERANCE,
    ROOM_CHANGE_TIME=ROOM_CHANGE_TIME
)

current_seq = tmp[tmp["scheduled_by_seq"] == 1].sort_values(
    ["day","shift","room","seq_in_block"]
).reset_index(drop=True)

# 3) Avaliar solução inicial do LS4 (AGORA SIM, sequenciada)
current_score, current_rooms_free, current_feas = \
    full_evaluation(current_seq)

prev_metrics = eval_components(
    current_seq, current_rooms_free, current_feas
)



# 2) Avaliar solução de partida (full_evaluation preserva order_hint porque não a remove)


best_score = current_score
best_enriched = current_enriched.copy()
best_seq = current_seq.copy()
best_rooms_free = current_rooms_free.copy()
best_feas = current_feas

print("Initial resequence score:", current_score)

N_ILS_ITER = 100

# Resetar contador para LS4
ils_iteration_counter = 0

for it in range(N_ILS_ITER):
    ils_iteration_counter += 1

    # --- 0) avaliar o estado atual (já sequenciado) ---
    # (current_seq tem de existir como expliquei acima)
    prev_score, prev_rooms, prev_feas = full_evaluation(current_seq)
    prev_metrics = eval_components(current_seq, prev_rooms, prev_feas)

    # --- 1) gerar vizinho mexendo só na order_hint ---
    neighbor_enriched, change_log = generate_neighbor_resequence(
        current_enriched, max_blocks_to_change=1, swaps_per_block=1
    )

    neighbor_assignments_ls4 = neighbor_enriched.drop(
        columns=[c for c in drop_ls4_cols if c in neighbor_enriched.columns]
    ).copy()

    # --- 2) sequenciar vizinho ---
    neighbor_seq = normalize_and_resequence(
        neighbor_assignments_ls4,
        df_patients,
        C_PER_SHIFT,
        CLEANUP,
        TOLERANCE,
        ROOM_CHANGE_TIME
    ).sort_values(["day","shift","room","seq_in_block"]).reset_index(drop=True)


    # --- 3) avaliar vizinho ---
    neigh_score, neigh_rooms_free, neigh_feas = full_evaluation(neighbor_seq)
    new_metrics = eval_components(neighbor_seq, neigh_rooms_free, neigh_feas)

    log_iteration("LS4_RESEQUENCE", ils_iteration_counter, new_metrics, accepted=(neigh_score > prev_score))

    # --- 4) aceitar (first improvement) ---
    if neigh_score > prev_score:
        current_enriched = neighbor_enriched.copy()
        current_assignments_ls4 = neighbor_assignments_ls4.copy()

        # IMPORTANT: atualizar também a versão sequenciada corrente
        current_seq = neighbor_seq.copy()
        current_score = neigh_score
        current_rooms_free = neigh_rooms_free
        current_feas = neigh_feas
        log_global_iteration(
            "LS4_RESEQUENCE",
            it,
            current_seq,
            current_rooms_free,
            current_feas
        )

        # LOG de melhorias
        nb = len(change_log)
        if nb >= 1:
            k = change_log[0]
            acao_curta = f"resequence: {nb} bloco(s), ex: room={k['room']} day={k['day']} shift={k['shift']}"
        else:
            acao_curta = "resequence: 0 blocos (n/a)"
        acao_detalhe = change_log
        log_improvement("LS4_RESEQUENCE", it, prev_metrics, new_metrics, acao_curta, acao_detalhe)

        # atualizar best
        if neigh_score > best_score:
            best_score = neigh_score
            best_enriched = neighbor_enriched.copy()
            best_seq = neighbor_seq.copy()
            best_rooms_free = neigh_rooms_free.copy()
            best_feas = neigh_feas

print("\n========== END OF ILS #4 ==========\n")


# === SOLUÇÃO FINAL DEPOIS DOS ILS (INCLUINDO RESEQUENCING) ===

# melhor solução encontrada pelo ILS3
assignments_enriched = best_enriched.copy()
assignments_seq_view = best_seq.copy()
# Recalcular métricas para a SOLUÇÃO FINAL (pós-LS4)
best_assignments_seq = assignments_seq_view.copy()
best_rooms_free = build_room_free_from_assignments(
    assignments=best_assignments_seq, df_rooms=df_rooms, C_PER_SHIFT=C_PER_SHIFT
)
best_feas = feasibility_metrics(
    assignments=best_assignments_seq,
    df_rooms=df_rooms,
    df_surgeons=df_surgeons,
    patients=df_patients,
    C_PER_SHIFT=C_PER_SHIFT
)
best_eval = evaluate_schedule(
    assignments=best_assignments_seq,
    patients=df_patients,
    rooms_free=best_rooms_free,
    excess_block_min=best_feas["excess_block_min"]
)
print(f"\nFINAL BEST SCORE: {best_eval['score']:.6f}")


df_room_free = best_rooms_free.copy()

# recomputar também liberdade dos cirurgiões para a solução final
df_surgeon_free = build_surgeon_free_from_assignments(
    assignments_seq_view,
    df_surgeons,
    C_PER_SHIFT
)

# e garantir que as variáveis "best_*" usadas no score final batem certo
best_assignments_seq = assignments_seq_view.copy()
best_rooms_free = df_room_free.copy()

# =========================================================
#                GVNS MAIN LOOP
# =========================================================

S_curr = assignments_seq_view.copy()
S_curr_score, _, _ = full_evaluation(S_curr)

k = 1  # shaking index

while k <= 2:

    print(f"\n========== GVNS | SHAKING {k} ==========")

    # ---------- SHAKING ----------
    S_shaken = shaking(
        S_curr,
        k,
        df_patients,
        df_rooms,
        df_surgeons,
        C_PER_SHIFT
    )

    S_local = normalize_and_sequence(
        S_shaken,
        df_patients,
        C_PER_SHIFT,
        CLEANUP,
        TOLERANCE,
        ROOM_CHANGE_TIME
    )
    

    base_score, _, _ = full_evaluation(S_local)
    print(base_score)
    # =================================================
    # MOVE 1 — LS2 (ADD ONLY)  — GVNS CORRETO
    # =================================================
    print("→ MOVE 1: ADD ONLY")

    # solução base do MOVE (NÃO MUDA durante o move)
    current = S_local.copy()
    current_score, current_rooms_free, current_feas = \
        full_evaluation(current)

    # melhor solução encontrada neste MOVE
    best_seq = current.copy()
    best_score = current_score
    best_rooms_free = current_rooms_free.copy()
    best_feas = current_feas

    N_ILS2_ITER = 50
    print("Initial add-only score:", current_score)

    for it in range(N_ILS2_ITER):
        ils_iteration_counter += 1

        # métricas da solução BASE (para logging)
        prev_metrics = eval_components(
            current,
            current_rooms_free,
            current_feas
        )

        neighbor, ids_added = generate_neighbor_add_only(
            current,
            df_patients,
            df_rooms,
            df_surgeons,
            C_PER_SHIFT,
            max_add=2
        )
        if not ids_added:
            continue

        # ---- sequenciar o vizinho ----
        neighbor_seq = normalize_and_sequence(
            neighbor,
            df_patients,
            C_PER_SHIFT,
            CLEANUP,
            TOLERANCE,
            ROOM_CHANGE_TIME
        ).sort_values(
            ["day", "shift", "room", "seq_in_block"]
        ).reset_index(drop=True)

        # ---- avaliar ----
        neigh_score, neigh_rooms_free, neigh_feas = \
            full_evaluation(neighbor_seq)

        new_metrics = eval_components(
            neighbor_seq,
            neigh_rooms_free,
            neigh_feas
        )

        # log_iteration(
        #     "ILS2_ADD_ONLY",
        #     ils_iteration_counter,
        #     new_metrics,
        #     accepted=(neigh_score > best_score)
        # )

        # ✅ APENAS atualizar o MELHOR do MOVE
        if neigh_score > best_score:
            best_score = neigh_score
            best_seq = neighbor_seq.copy()
            best_rooms_free = neigh_rooms_free.copy()
            best_feas = neigh_feas

            acao_curta = f"add-only: +{len(ids_added)}"
            acao_detalhe = {"adicionados": ids_added}
            log_improvement(
                "ILS2_ADD_ONLY",
                it,
                prev_metrics,
                new_metrics,
                acao_curta,
                acao_detalhe
            )

    # sÓ AQUI o MOVE termina e S_local é atualizado
    S_local = best_seq.copy()

    # =================================================
    # MOVE 2 — LS1 (SWAP)  — GVNS CORRETO
    # =================================================
    print("→ MOVE 2: SWAP")
    
    # solução base do MOVE (NÃO MUDA durante o move)
    current = S_local.copy()
    current_score, current_rooms_free, current_feas = \
        full_evaluation(current)
    
    # melhor solução encontrada neste MOVE
    best_seq = current.copy()
    best_score = current_score
    best_rooms_free = current_rooms_free.copy()
    best_feas = current_feas
    
    N_ILS_ITER = 100
    MAX_SWAP_OUT = 2
    MAX_SWAP_IN  = 2
    
    print("Initial swap score:", current_score)
    
    for it in range(N_ILS_ITER):
        ils_iteration_counter += 1
    
        # métricas da solução BASE (para logging)
        prev_metrics = eval_components(
            current,
            current_rooms_free,
            current_feas
        )
    
        # -------- 1) GERAR VIZINHO --------
        neighbor_struct, ids_out, ids_in = generate_neighbor_swap(
            current,
            df_patients,
            df_rooms,
            df_surgeons,
            C_PER_SHIFT,
            max_swap_out=MAX_SWAP_OUT,
            max_swap_in=MAX_SWAP_IN
        )
    
        # -------- 2) SEQUENCIAR --------
        neighbor_seq = normalize_and_sequence(
            neighbor_struct,
            df_patients,
            C_PER_SHIFT,
            CLEANUP,
            TOLERANCE,
            ROOM_CHANGE_TIME
        ).sort_values(
            ["day", "shift", "room", "seq_in_block"]
        ).reset_index(drop=True)
    
        # -------- 3) AVALIAR --------
        neigh_score, neigh_rooms_free, neigh_feas = \
            full_evaluation(neighbor_seq)
    
        new_metrics = eval_components(
            neighbor_seq,
            neigh_rooms_free,
            neigh_feas
        )
    
        # log_iteration(
        #     "ILS1_SWAP",
        #     ils_iteration_counter,
        #     new_metrics,
        #     accepted=(neigh_score > best_score)
        # )
    
        # ✅ APENAS atualizar o MELHOR do MOVE
        if neigh_score > best_score:
            best_score = neigh_score
            best_seq = neighbor_seq.copy()
            best_rooms_free = neigh_rooms_free.copy()
            best_feas = neigh_feas
    
            acao_curta = f"swap: -{len(ids_out)} +{len(ids_in)}"
            acao_detalhe = {
                "removed": ids_out,
                "added": ids_in
            }
    
            log_improvement(
                "ILS1_SWAP",
                it,
                prev_metrics,
                new_metrics,
                acao_curta,
                acao_detalhe
            )
    
    # SÓ AQUI o MOVE termina e S_local é atualizado
    S_local = best_seq.copy()
    
    # =================================================
    # MOVE 3 — LS3 (CROSS ROOM)  — GVNS CORRETO
    # =================================================
    print("→ MOVE 3: CROSS ROOM")
    
    # solução base do MOVE (NÃO MUDA durante o move)
    current = S_local.copy()
    current_score, current_rooms_free, current_feas = \
        full_evaluation(current)
    
    # melhor solução encontrada neste MOVE
    best_seq = current.copy()
    best_score = current_score
    best_rooms_free = current_rooms_free.copy()
    best_feas = current_feas
    
    print("Initial score (LS3):", current_score)
    
    N_ILS4_ITER = 50
    
    for it in range(N_ILS4_ITER):
        ils_iteration_counter += 1
    
        # métricas da solução BASE (para logging)
        prev_metrics = eval_components(
            current,
            current_rooms_free,
            current_feas
        )
    
        # -------- 1) GERAR VIZINHO --------
        neighbor, swap_info = generate_neighbor_cross_room_swap(current)
        if swap_info is None:
            continue
    
        # -------- 2) SEQUENCIAR --------
        neighbor_seq = normalize_and_sequence(
            neighbor,
            df_patients,
            C_PER_SHIFT,
            CLEANUP,
            TOLERANCE,
            ROOM_CHANGE_TIME
        ).sort_values(
            ["day", "shift", "room", "seq_in_block"]
        ).reset_index(drop=True)
    
        # -------- 3) AVALIAR --------
        neigh_score, neigh_rooms_free, neigh_feas = \
            full_evaluation(neighbor_seq)
    
        new_metrics = eval_components(
            neighbor_seq,
            neigh_rooms_free,
            neigh_feas
        )
    
        # log_iteration(
        #     "LS3_CROSS_ROOM",
        #     ils_iteration_counter,
        #     new_metrics,
        #     accepted=(neigh_score > best_score)
        # )
    
        # ✅ APENAS atualizar o MELHOR do MOVE
        if neigh_score > best_score:
            best_score = neigh_score
            best_seq = neighbor_seq.copy()
            best_rooms_free = neigh_rooms_free.copy()
            best_feas = neigh_feas
    
            acao_curta = (
                f"cross-room swap: day={swap_info['day']} "
                f"shift={swap_info['shift']} "
                f"{swap_info['roomA']}↔{swap_info['roomB']}"
            )
            acao_detalhe = swap_info
    
            log_improvement(
                "LS3_CROSS_ROOM",
                it,
                prev_metrics,
                new_metrics,
                acao_curta,
                acao_detalhe
            )
    
    #  SÓ AQUI o MOVE termina e S_local é atualizado
    S_local = best_seq.copy()

    # =================================================
    # MOVE 4 — LS4 (RESEQUENCE)  — GVNS CORRETO
    # =================================================
    print("→ MOVE 4: RESEQUENCE")
    
    # ====== BASE DO MOVE (NÃO MUDA DURANTE O MOVE) ======
    current_enriched = S_local.copy()
    
    # Garantir colunas clínicas
    if not {'duration', 'priority', 'waiting'}.issubset(current_enriched.columns):
        current_enriched = current_enriched.merge(
            df_patients[['patient_id', 'duration', 'priority', 'waiting']],
            on='patient_id',
            how='left'
        )
    
    # Garantir colunas de sequenciamento (se faltarem, sequenciar uma vez)
    if not {'start_min', 'end_min', 'seq_in_block', 'scheduled_by_seq'}.issubset(current_enriched.columns):
        current_enriched = sequence_global_by_surgeon(
            current_enriched,
            C_PER_SHIFT=C_PER_SHIFT,
            CLEANUP=CLEANUP,
            TOLERANCE=TOLERANCE,
            ROOM_CHANGE_TIME=ROOM_CHANGE_TIME
        )
    
    # Garantir order_hint base (pista)
    current_enriched['order_hint'] = current_enriched['seq_in_block']
    
    # 1) Criar assignments "limpo" (MANTER order_hint)
    drop_ls4_cols = [
        "duration", "priority", "waiting",
        "start_min", "end_min", "seq_in_block", "scheduled_by_seq"
    ]
    current_assignments_ls4 = current_enriched.drop(
        columns=[c for c in drop_ls4_cols if c in current_enriched.columns]
    ).copy()
    
    # 2) Sequenciar solução inicial do LS4 (para ter score base)
    current_seq = normalize_and_resequence(
        current_assignments_ls4,
        df_patients,
        C_PER_SHIFT,
        CLEANUP,
        TOLERANCE,
        ROOM_CHANGE_TIME
    ).sort_values(["day","shift","room","seq_in_block"]).reset_index(drop=True)
    
    current_score, current_rooms_free, current_feas = full_evaluation(current_seq)


    
    print("Initial resequence score:", current_score)
    
    # ====== MELHOR DO MOVE ======
    best_score = current_score
    best_seq = current_seq.copy()
    best_rooms_free = current_rooms_free.copy()
    best_feas = current_feas
    
    N_ILS_ITER = 30
    
    for it in range(N_ILS_ITER):
        ils_iteration_counter += 1
    
        # métricas da BASE (para logging coerente)
        prev_metrics = eval_components(current_seq, current_rooms_free, current_feas)
    
        # --- 1) gerar vizinho mexendo só na order_hint (a partir da BASE FIXA do move) ---
        neighbor_enriched, change_log = generate_neighbor_resequence(
            current_enriched, max_blocks_to_change=1, swaps_per_block=1
        )
    
        neighbor_assignments_ls4 = neighbor_enriched.drop(
            columns=[c for c in drop_ls4_cols if c in neighbor_enriched.columns]
        ).copy()
    
        # --- 2) resequence (usa order_hint) ---
        neighbor_seq = normalize_and_resequence(
            neighbor_assignments_ls4,
            df_patients,
            C_PER_SHIFT,
            CLEANUP,
            TOLERANCE,
            ROOM_CHANGE_TIME
        ).sort_values(["day","shift","room","seq_in_block"]).reset_index(drop=True)
    
        # --- 3) avaliar vizinho ---
        neigh_score, neigh_rooms_free, neigh_feas = full_evaluation(neighbor_seq)
        new_metrics = eval_components(neighbor_seq, neigh_rooms_free, neigh_feas)
    
        # ✅ accepted = melhora o MELHOR DO MOVE (não é “aceitação durante o move”)
        log_iteration("LS4_RESEQUENCE", it, new_metrics, accepted=(neigh_score > best_score))
    
        # ✅ só guarda melhor do move
        if neigh_score > best_score:
            best_score = neigh_score
            best_seq = neighbor_seq.copy()
            best_rooms_free = neigh_rooms_free.copy()
            best_feas = neigh_feas
    
            nb = len(change_log)
            if nb >= 1:
                example = change_log[0]
                acao_curta = (
                    f"resequence: {nb} bloco(s), "
                    f"ex: room={example['room']} day={example['day']} shift={example['shift']}"
                )
            else:
                acao_curta = "resequence: 0 blocos (n/a)"
    
            log_improvement("LS4_RESEQUENCE", it, prev_metrics, new_metrics, acao_curta, change_log)
    
    # ⬅️ SÓ AQUI o MOVE termina e S_local é atualizado
    S_local = best_seq.copy()
    
    

    # =================================================
    # DECISÃO GVNS
    # =================================================
    final_score, _, _ = full_evaluation(S_local)

    print(f"\nCompare:")
    print(f"  before shaking: {S_curr_score:.6f}")
    print(f"  after GVNS:     {final_score:.6f}")

    if final_score > S_curr_score:
        print("✔ Improvement → back to SHAKING 1")
        S_curr = S_local.copy()
        S_curr_score = final_score
        k = 1
        tmp_score, tmp_rooms, tmp_feas = full_evaluation(S_curr)
        log_global_iteration("GVNS_ACCEPT", k, S_curr, tmp_rooms, tmp_feas)
    else:
        if k == 1:
            print("✘ No improvement → SHAKING 2")
            k = 2
        else:
            print("✘ No improvement in SHAKING 2 → END GVNS")
            break



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
    initial_assignments_enriched.to_excel(writer, sheet_name="Assignments_Initial", index=False)
    initial_assignments_seq.to_excel(writer, sheet_name="Assignments_Sequenced_Initial", index=False)
    initial_rooms_free.to_excel(writer, sheet_name="Rooms_Free_Initial", index=False)
    initial_surgeons_free.to_excel(writer, sheet_name="Surgeons_Free_Initial", index=False)

    # ============ RESULTS – SOLUÇÃO FINAL (ILS) ============
    assignments_enriched.to_excel(writer, sheet_name="Assignments_ILS", index=False)
    assignments_seq_view.to_excel(writer, sheet_name="Assignments_Sequenced_ILS", index=False)
    final_blocks.to_excel(writer, sheet_name="Blocks_FinalState_ILS", index=False)
    rooms_free.to_excel(writer, sheet_name="Rooms_Free_ILS", index=False)
    surgeons_free.to_excel(writer, sheet_name="Surgeons_Free_ILS", index=False)

    # KPIs (solução final)
    kpi_global.to_excel(writer, sheet_name="KPI_Global_ILS", index=False)
    kpi_day_rooms.to_excel(writer, sheet_name="KPI_PerDay_ILS", index=False)
    kpi_room.to_excel(writer, sheet_name="KPI_PerRoom_ILS", index=False)
    kpi_surgeon.to_excel(writer, sheet_name="KPI_PerSurgeon_ILS", index=False)

    # Unassigned (solução final)
    unassigned_patients.to_excel(writer, sheet_name="Unassigned_ILS", index=False)

    # ======== SENSITIVITY LOG (melhorias aceites) ========
    if improvement_log:
        df_score_log = pd.DataFrame(improvement_log).sort_values(["fase", "iteracao"])
    else:
        df_score_log = pd.DataFrame([{"info": "Nenhuma melhoria registada"}])
    df_score_log.to_excel(writer, sheet_name="Score_Improvement_Log", index=False)

    # ======== ITERATIONS LOG (todas as iterações, só termos do score) ========
    if iteration_log:
        df_iter_log = pd.DataFrame(iteration_log).sort_values(["fase", "iteracao"])
    else:
        df_iter_log = pd.DataFrame([{"info": "Sem iterações registadas"}])
    df_iter_log.to_excel(writer, sheet_name="Iterations_Log", index=False)  
    
    # ======== GLOBAL ITER LOG (snapshots das soluções aceites) ========
    if global_iter_log:
        df_global = pd.DataFrame(global_iter_log)
    else:
        df_global = pd.DataFrame([{"info": "Sem snapshots globais"}])
    
    df_global.to_excel(writer, sheet_name="Global_Iterations_Log", index=False)


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
# RUN FEASIBILITY + EVALUATION ON final SOLUTION 
# --------------------------------------------
feas = feasibility_metrics(
    assignments=assignments_seq_view,   
    df_rooms=df_rooms,
    df_surgeons=df_surgeons,
    patients=df_patients,
    C_PER_SHIFT=C_PER_SHIFT
)


# ------- SCORE initial -------

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

