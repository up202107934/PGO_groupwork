int NumberPatients = ...;
int NumberOfRooms = ...;
int NumberSurgeons = ...;
int NumberOfDays = ...;

range Patients = 1..NumberPatients;
range Rooms = 1..NumberOfRooms;
range Surgeons = 1..NumberSurgeons;
range Days = 1..NumberOfDays;
range Shifts = 1..2;
range Priorities = 1..4;

int Tmax[Priorities] = [270,90,15,3];


Prioridade 0: 270 dias
Prioridade 1: 90 dias
Prioridade 2: 15 dias
Prioridade 3: 3 dias

Quatro níveis de prioridade possíveis. Uma cirurgia classificada como Urgência Diferida deverá realizar-se em 72 horas. Como Muito Prioritária deverá ser realizada dentro de um período de 15 dias. A cirurgia Prioritária deverá acontecer num período de 2 meses. A cirurgia classificada como Normal deverá ocorrer dentro de um período de 1 ano.

int capacity = 360; // minutes
int clean = 17; // minutes //setup na sala do bloco operatório antes de uma cirurgia

int Duration[Patients] = ...;
int Waiting[Patients] = ...;
int Priority[Patients] = ...;
int Surgeon[Patients] = ...;

int SurgeonAvailability[Surgeons][Days][Shifts] = ...;
int BlockAvailability[Days][Rooms][Shifts] = ...; // to a certain specialty

