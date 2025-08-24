import subprocess
from datetime import datetime

# Configurazione iniziale
MAX_TURNS = 100

DEPTH = 0
TIME_TOTAL_SEC = 5

TIME_H = TIME_TOTAL_SEC // 3600
TIME_M = TIME_TOTAL_SEC // 60
TIME_S = TIME_TOTAL_SEC % 60

NUM_TOURNAMENTS = 3

OK = "ok\n"

# Variabili globali per i path
engine_paths = [
    "./nokamute",
    "./strivecpp_undo_tt",
    "./strivecpp_1",
    "./strivecpp_3",
]

mode = "TIME"
folder_path = ""


def send(p: subprocess.Popen, command: str) -> str:
    p.stdin.write(command + "\n")
    p.stdin.flush()
    return read_all(p)

def readuntil(p: subprocess.Popen, delim: str) -> str:
    output = []
    while True:
        line = p.stdout.readline()
        if not line:
            break
        output.append(line.strip())
        if line.endswith(delim):
            break
    return "\n".join(output)

def read_all(p: subprocess.Popen) -> str:
    return readuntil(p, OK)

def start_process(path, args=[]) -> subprocess.Popen:
    return subprocess.Popen(
        [path] + args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True,
    )

def end_process(p: subprocess.Popen) -> None:
    p.stdin.close()
    p.stdout.close()
    p.stderr.close()
    p.kill()

def play_step(p1: subprocess.Popen, p2: subprocess.Popen) -> str:
    if DEPTH != 0:
        move = send(p1, f"bestmove depth {DEPTH}")
    else:
        move = send(p1, f"bestmove time {TIME_H:02}:{TIME_M:02}:{TIME_S:02}")
    move = move.strip().split("\n")[0]
    #print(f"[Player] plays: {move}")
    send(p1, f"play {move}")
    return send(p2, f"play {move}")

def check_end_game(out: str) -> bool:
    return "InProgress" != out.split(";")[1]

def produce_file(info):
    res = f'[GameType "{info[0]}"]\n[Result "{info[1]}"]\n'
    for i in enumerate(info[3:],1):
        res += f"{i[0]}. {i[1]}\n"
    return res

def start_game(path1, path2):
    
    print(f"Starting interaction with {path1}...")
    player1 = start_process(path1)
    read_all(player1)
    print(f"Starting interaction with {path2}...")
    player2 = start_process(path2)
    read_all(player2)

    send(player1, "newgame Base+MLP")
    send(player2, "newgame Base+MLP")

    info = ["Base+MLP", "Draw"]  # Valore di default in caso di pareggio
    for _ in range(MAX_TURNS//2):
        out = play_step(player1, player2)
        if check_end_game(out):
            print(out)
            info = out.split("\n")[0].split(";")
            break

        out = play_step(player2, player1)
        if check_end_game(out):
            print(out)
            info = out.split("\n")[0].split(";")
            break

    send(player1, "exit")
    send(player2, "exit")
    end_process(player1)
    end_process(player2)
    print("Game over.")
    # Stampa del file di output nella cartella di outut
    if folder_path:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        output_file = f"{folder_path}/{timestamp}.pgn"
        with open(output_file, "w") as f:
            f.write(produce_file(info))
        print(f"File salvato in {output_file}")
    else:
        print("Nessuna cartella selezionata per il salvataggio del file.")
    
    return info[1] # risultato della partita

def start_tournament():
    global MAX_TURNS, engine_paths
    if len(engine_paths) < 2:
        print("Selezionare almeno due file di engine.")
        return
    
    # Dizionario per tenere traccia dei risultati
    results = {path: 0 for path in engine_paths}

    for _ in range(NUM_TOURNAMENTS):
    
        # Girone di andata
        for i in range(len(engine_paths)):
            for j in range(i + 1, len(engine_paths)):
                print(f"Starting match between {engine_paths[i].split('/')[-1]} and {engine_paths[j].split('/')[-1]}")
                result = start_game(engine_paths[i], engine_paths[j])
                if result == "WhiteWins":
                    results[engine_paths[i]] += 3
                elif result == "BlackWins":
                    results[engine_paths[j]] += 3
                else:
                    results[engine_paths[i]] += 1
                    results[engine_paths[j]] += 1

        # Girone di ritorno
        engine_paths = engine_paths[::-1]

        for i in range(len(engine_paths)):
            for j in range(i + 1, len(engine_paths)):
                print(f"Starting match between {engine_paths[i].split('/')[-1]} and {engine_paths[j].split('/')[-1]}")
                result = start_game(engine_paths[i], engine_paths[j])
                if result == "WhiteWins":
                    results[engine_paths[i]] += 3
                elif result == "BlackWins":
                    results[engine_paths[j]] += 3
                else:
                    results[engine_paths[i]] += 1
                    results[engine_paths[j]] += 1


    # Ordina i risultati
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    # Stampa i risultati finali
    print("Tournament Results:")
    print("+-------------------------+--------+")
    print("| Engine Name             | Points |")
    print("+-------------------------+--------+")
    for engine, score in sorted_results:
        engine_name = engine.split("/")[-1]
        print(f"| {engine_name[:23]:<23} | {score:<6} |")
    print("+-------------------------+--------+")


start_tournament()
