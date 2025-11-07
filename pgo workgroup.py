# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# scheduling_weights.py
# -------------------------------------------------------
# Computes W(i) for each patient using alpha-weighted terms
# and prints the top patients based on their heuristic score
# -------------------------------------------------------

from pathlib import Path
import re, ast

DATA_FILE = "instance_cv_30.dat"

# ---- α coefficients ----
ALPHA1 = 0.45  # priority
ALPHA2 = 0.15  # waited days
ALPHA3 = 0.35  # deadline closeness
ALPHA4 = 0.05  # feasible blocks
# ------------------------


# -------------------------------------------------------
#                Load instance data
# -------------------------------------------------------
def load_instance(filename):
    text = Path(filename).read_text()

    def get_int(name, alt_name=None):
        m = re.search(rf'int {name}\s*=\s*(\d+)', text)
        if not m and alt_name:
            m = re.search(rf'int {alt_name}\s*=\s*(\d+)', text)
        if not m:
            raise ValueError(f"Could not find integer '{name}' in {filename}")
        return int(m.group(1))

    def get_array(name):
        m = re.search(rf'{name}\s*=\s*\[(.*?)\];', text, re.DOTALL)
        if not m:
            raise ValueError(f"Could not find array '{name}' in {filename}")
        raw = m.group(1).replace("\n", "")
        return [int(x) for x in raw.split(",") if x.strip()]

    data = {}
    data["NumberPatients"] = get_int("NumberPatients")
    data["NumberOfRooms"] = get_int("NumberOfRooms")
    data["NumberSurgeons"] = get_int("NumberSurgeons", alt_name="NumberOfSurgeons")
    data["NumberOfDays"] = get_int("NumberOfDays")

    data["Duration"] = get_array("Duration")
    data["Priority"] = get_array("Priority")
    data["WaitDays"] = get_array("Waiting")
    data["Surgeon"] = get_array("Surgeon")

    # 3D arrays
    block_match = re.search(
        r'BlockAvailability\s*=\s*(\[\s*\[\s*\[.*?\]\s*\]\s*\]);',
        text,
        re.DOTALL,
    )
    if not block_match:
        raise ValueError("Could not find BlockAvailability in file.")
    data["BlockAvailability"] = ast.literal_eval(block_match.group(1))

    surg_match = re.search(
        r'SurgeonAvailability\s*=\s*(\[\s*\[\s*\[.*?\]\s*\]\s*\]);',
        text,
        re.DOTALL,
    )
    if not surg_match:
        raise ValueError("Could not find SurgeonAvailability in file.")
    data["SurgeonAvailability"] = ast.literal_eval(surg_match.group(1))

    return data


# -------------------------------------------------------
#             Feasible blocks counter
# -------------------------------------------------------
def count_feasible_blocks(data, patient_idx):
    """Count (day, room, shift) slots feasible for the patient’s surgeon."""
    surgeon_id = data["Surgeon"][patient_idx] - 1
    block_av = data["BlockAvailability"]     # [day][room][shift]
    surg_av = data["SurgeonAvailability"]    # [surgeon][day][shift]

    num_days = len(block_av)
    num_rooms = len(block_av[0])
    num_shifts = len(block_av[0][0])

    cnt = 0
    for d in range(num_days):
        for r in range(num_rooms):
            for s in range(num_shifts):
                if block_av[d][r][s] == 1 and surg_av[surgeon_id][d][s] == 1:
                    cnt += 1
    return cnt


# -------------------------------------------------------
#                Compute W(i)
# -------------------------------------------------------
def compute_weights(data):
    n = data["NumberPatients"]
    pr = data["Priority"]
    wait = data["WaitDays"]

    maxP = max(pr) or 1
    maxWait = max(wait) or 1

    results = []

    for i in range(n):
        priority_i = pr[i]
        waited_i = wait[i]

        # ---- DEADLINE RULES ----
        if priority_i == 2:          # high
            deadline_limit = 15
        elif priority_i == 1:        # medium
            deadline_limit = 270
        else:                        # priority 0 -> no deadline
            deadline_limit = None
        # ------------------------

        # ---- Compute deadline term ----
        if deadline_limit is None:
            deadline_term = 0.0
            days_left = None
        else:
            days_left = max(0, deadline_limit - waited_i)
            # normalized for weighting only
            deadline_term = 1 - (days_left / deadline_limit)
        # -------------------------------

        feas = count_feasible_blocks(data, i)

        # normalized terms
        term1 = priority_i / maxP
        term2 = waited_i / maxWait
        term3 = deadline_term
        term4 = 1 / (1 + feas)

        W = (
            ALPHA1 * term1
            + ALPHA2 * term2
            + ALPHA3 * term3
            + ALPHA4 * term4
        )

        results.append(
            {
                "patient": i + 1,
                "W": W,
                "priority": priority_i,
                "wait": waited_i,
                "deadline_limit": deadline_limit,
                "deadline_term": deadline_term,
                "days_left": days_left,
                "feasible_blocks": feas,
            }
        )

    # highest W first
    results.sort(key=lambda x: x["W"], reverse=True)
    return results


# -------------------------------------------------------
#                     MAIN
# -------------------------------------------------------
def main():
    data = load_instance(DATA_FILE)
    ranked = compute_weights(data)

    print("Top 500 patients (weighted score W(i)):\n")
    print("Patient |   W(i)   | P | Waited | Deadline | Days_Left | FeasBlocks")
    print("-" * 67)

    for row in ranked[:500]:
        if row["deadline_limit"] is None:
            ddl = "None"
            left = "-"
        else:
            ddl = f"{row['deadline_limit']:3d}"
            left = f"{row['days_left']:3d}"

        print(
            f"P{row['patient']:4d} | "
            f"W={row['W']:.4f} | "
            f"P={row['priority']} | "
            f"waited={row['wait']:4d} | "
            f"ddl={ddl:>5s} | "
            f"left={left:>5s} | "
            f"feas={row['feasible_blocks']:2d}"
        )


if __name__ == "__main__":
    main()

