from pathlib import Path
import re
import ast
import random

# Se quiseres resultados reprodutíveis, descomenta esta linha:
# random.seed(42)

# ============================================================
# 1) Funções genéricas para ler o .dat
# ============================================================

def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def get_int_from_text(text, name, alt_name=None):
    """
    Procura 'int name = X' no texto.
    Se não encontrar e alt_name for dado, tenta 'int alt_name = X'.
    """
    m = re.search(rf'int\s+{name}\s*=\s*(\d+)', text)
    if not m and alt_name:
        m = re.search(rf'int\s+{alt_name}\s*=\s*(\d+)', text)
    if not m:
        raise ValueError(f"Integer '{name}' not found in instance file.")
    return int(m.group(1))


def get_array_from_text(text, name):
    """
    Procura 'name = [ ... ];' no texto e devolve o array como objeto Python.
    """
    m = re.search(rf'{name}\s*=\s*(\[[\s\S]*?\])\s*;', text, re.DOTALL)
    if not m:
        raise ValueError(f"Array '{name}' not found in instance file.")
    return ast.literal_eval(m.group(1))


def load_instance(path):
    """
    Lê um ficheiro .dat no formato usado pelo teu projeto e devolve um dicionário.
    """
    txt = read_text(path)

    data = {}
    data["NumberPatients"] = get_int_from_text(txt, "NumberPatients")
    data["NumberOfRooms"]  = get_int_from_text(txt, "NumberOfRooms")
    # Nos teus códigos usas 'NumberSurgeons' com alt_name 'NumberOfSurgeons'
    data["NumberSurgeons"] = get_int_from_text(txt, "NumberSurgeons", alt_name="NumberOfSurgeons")
    data["NumberOfDays"]   = get_int_from_text(txt, "NumberOfDays")

    data["Duration"]            = get_array_from_text(txt, "Duration")
    data["Priority"]            = get_array_from_text(txt, "Priority")
    data["Waiting"]             = get_array_from_text(txt, "Waiting")
    data["Surgeon"]             = get_array_from_text(txt, "Surgeon")
    data["BlockAvailability"]   = get_array_from_text(txt, "BlockAvailability")
    data["SurgeonAvailability"] = get_array_from_text(txt, "SurgeonAvailability")

    return data

# ============================================================
# 2) Construir distribuições globais a partir de VÁRIAS instâncias
# ============================================================

BASE_FILES = [
    "Instance_C1_30.dat",
    "Instance_C2_30.dat",
    "Instance_C3_30.dat",
    "Instance_CAT_30.dat",
    "Instance_CMF_30.dat",
]

def load_all_bases(file_list):
    bases = []
    for p in file_list:
        path = Path(p)
        if path.exists():
            bases.append(load_instance(path))
        else:
            print(f"[WARNING] Base file not found: {p}")
    if not bases:
        raise RuntimeError("No base instances loaded. Check BASE_FILES paths.")
    return bases


def build_global_distributions(bases):
    """
    Junta todas as listas de Duration, Priority, Waiting e Surgeon
    para amostrar a partir de TODAS as instâncias reais, não só de uma.
    """
    all_durations  = []
    all_priorities = []
    all_waitings   = []
    all_surgeons   = []

    for b in bases:
        all_durations  += b["Duration"]
        all_priorities += b["Priority"]
        all_waitings   += b["Waiting"]
        all_surgeons   += b["Surgeon"]

    return {
        "durations":  all_durations,
        "priorities": all_priorities,
        "waitings":   all_waitings,
        "surgeons":   all_surgeons,
    }


def estimate_block_open_probability(bases):
    """
    Estima a probabilidade média de um bloco (room, day, shift) estar aberto.
    BlockAvailability = [day][room][shift]
    """
    open_count = 0
    total_count = 0

    for b in bases:
        bav = b["BlockAvailability"]
        n_days = len(bav)
        n_rooms = len(bav[0])
        for d in range(n_days):
            for r in range(n_rooms):
                for s in range(2):  # AM/PM
                    total_count += 1
                    if bav[d][r][s] == 1:
                        open_count += 1

    if total_count == 0:
        return 0.8  # fallback
    return open_count / total_count


def estimate_surgeon_availability_probability(bases):
    """
    Estima a probabilidade média de um cirurgião estar disponível
    num dado (day, shift).
    SurgeonAvailability = [surgeon][day][shift]
    """
    avail_count = 0
    total = 0

    for b in bases:
        sav = b["SurgeonAvailability"]
        n_surg = len(sav)
        n_days = len(sav[0])
        for s in range(n_surg):
            for d in range(n_days):
                for sh in range(2):  # AM/PM
                    total += 1
                    if sav[s][d][sh] == 1:
                        avail_count += 1

    if total == 0:
        return 0.7  # fallback
    return avail_count / total

# ============================================================
# 3) Gerar nova instância a partir das distribuições globais
# ============================================================

def generate_instance_from_many(bases, global_dist, p_block_open, p_surg_av, n_patients=None):
    """
    Gera uma nova instância:
      - nº de pacientes aleatório entre [min, max] das bases (se n_patients=None)
      - sampling de Duration, Priority, Waiting, Surgeon de distribuições globais
      - BlockAvailability e SurgeonAvailability gerados aleatoriamente com
        probabilidades p_block_open e p_surg_av.
    """
    # usar a primeira base como referência estrutural (nº salas, cirurgiões, dias)
    ref = bases[0]
    n_rooms    = ref["NumberOfRooms"]
    n_surgeons = ref["NumberSurgeons"]
    n_days     = ref["NumberOfDays"]

    # nº de pacientes
    if n_patients is None:
        n_min = min(b["NumberPatients"] for b in bases)
        n_max = max(b["NumberPatients"] for b in bases)
        n_patients = random.randint(n_min, n_max)

    # pacientes: sampling global
    dur  = random.choices(global_dist["durations"],  k=n_patients)
    prio = random.choices(global_dist["priorities"], k=n_patients)
    wait = random.choices(global_dist["waitings"],   k=n_patients)
    surg = random.choices(global_dist["surgeons"],   k=n_patients)

    # BlockAvailability: [day][room][shift]
    new_block_av = [
        [
            [1 if random.random() < p_block_open else 0 for _sh in range(2)]
            for _r in range(n_rooms)
        ]
        for _d in range(n_days)
    ]

    # SurgeonAvailability: [surgeon][day][shift]
    new_surg_av = [
        [
            [1 if random.random() < p_surg_av else 0 for _sh in range(2)]
            for _d in range(n_days)
        ]
        for _s in range(n_surgeons)
    ]

    new_data = {
        "NumberPatients": n_patients,
        "NumberOfRooms":  n_rooms,
        "NumberSurgeons": n_surgeons,
        "NumberOfDays":   n_days,
        "Duration":       dur,
        "Priority":       prio,
        "Waiting":        wait,
        "Surgeon":        surg,
        "BlockAvailability":   new_block_av,
        "SurgeonAvailability": new_surg_av,
    }
    return new_data

# ============================================================
# 4) Escrever .dat no mesmo formato
# ============================================================

def format_list(lst):
    return "[" + ",".join(str(x) for x in lst) + "]"


def format_3d_array(arr):
    # imprime tipo [[[...]]] sem espaços
    return str(arr).replace(" ", "")


def write_dat_file(data, path):
    lines = []
    lines.append(f"int NumberPatients = {data['NumberPatients']}")
    lines.append(f"int NumberOfRooms = {data['NumberOfRooms']}")
    lines.append(f"int NumberSurgeons = {data['NumberSurgeons']}")
    lines.append(f"int NumberOfDays = {data['NumberOfDays']}")
    lines.append(f"Duration = {format_list(data['Duration'])};")
    lines.append(f"Priority = {format_list(data['Priority'])};")
    lines.append(f"Waiting = {format_list(data['Waiting'])};")
    lines.append(f"Surgeon = {format_list(data['Surgeon'])};")
    lines.append(f"BlockAvailability = {format_3d_array(data['BlockAvailability'])};")
    lines.append(f"SurgeonAvailability = {format_3d_array(data['SurgeonAvailability'])};")

    txt = "\n".join(lines) + "\n"
    Path(path).write_text(txt, encoding="utf-8")


# ============================================================
# 5) Exemplo de uso
# ============================================================

if __name__ == "__main__":
    bases = load_all_bases(BASE_FILES)
    global_dist = build_global_distributions(bases)
    p_block_open = estimate_block_open_probability(bases)
    p_surg_av    = estimate_surgeon_availability_probability(bases)

    print(f"Estimated block-open probability: {p_block_open:.3f}")
    print(f"Estimated surgeon-availability probability: {p_surg_av:.3f}")

    N_INSTANCES = 5  # muda aqui se quiseres mais

    for i in range(1, N_INSTANCES + 1):
        new_data = generate_instance_from_many(
            bases=bases,
            global_dist=global_dist,
            p_block_open=p_block_open,
            p_surg_av=p_surg_av,
            n_patients=None  # deixa None para escolher aleatório entre [min, max]
        )
        out_path = f"Instance_GEN_{i}.dat"
        write_dat_file(new_data, out_path)
        print(f"Written {out_path}")
