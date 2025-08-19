import subprocess
# import tkinter as tk
# from tkinter import filedialog

# Configurazione iniziale
NUMBER_OF_GAMES = 3
MAX_TURNS = 100
DEPTH = 0
TIME_TOTAL_SEC = 5
TIME_H = TIME_TOTAL_SEC // 3600
TIME_M = TIME_TOTAL_SEC // 60
TIME_S = TIME_TOTAL_SEC % 60
OK = "ok\n"

# Variabili globali per i path
mzinga_path = "/home/ubuntu/strhive-to-fly/strivecpp_undo_tt"
other_path = "/home/ubuntu/strhive-to-fly/cpp/build/Release/strivecpp"
mode = "DEPTH"

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

def start_process(path) -> subprocess.Popen:
    return subprocess.Popen(
        [path],
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
    print(f"[Player] plays: {move}", flush=True)
    send(p1, f"play {move}")
    return send(p2, f"play {move}")

def check_end_game(out: str) -> bool:
    return "InProgress" != out.split(";")[1]

def start_game():
    global MAX_TURNS, NUMBER_OF_GAMES

    if not mzinga_path or not other_path:
        print("Selezionare entrambi i file prima di avviare il gioco.")
        return
    
    # set_mode()
    
    print(f"Starting interaction with {mzinga_path.split("/")[-1]}...")
    mzinga = start_process(mzinga_path)
    read_all(mzinga)
    print(f"Starting interaction with {other_path.split("/")[-1]}...")
    other = start_process(other_path)
    read_all(other)

    whithe_wins = 0
    black_wins = 0
    draw = 0

    for i in range(NUMBER_OF_GAMES):
        print(f"Starting game {i + 1}/{NUMBER_OF_GAMES}...")
        send(mzinga, "newgame Base+MLP")
        send(other, "newgame Base+MLP")

        for _ in range(MAX_TURNS):
            out = play_step(mzinga, other)
            if check_end_game(out):
                if "WhiteWins" in out:
                    whithe_wins += 1
                elif "BlackWins" in out:
                    black_wins += 1
                else:
                    draw += 1
                print(out)
                break

            out = play_step(other, mzinga)
            if check_end_game(out):
                if "WhiteWins" in out:
                    whithe_wins += 1
                elif "BlackWins" in out:
                    black_wins += 1
                else:
                    draw += 1
                print(out)
                break

    send(mzinga, "exit")
    send(other, "exit")
    end_process(mzinga)
    end_process(other)
    print("--------- RESULTS ---------")
    print(f"White wins: {whithe_wins}")
    print(f"Black wins: {black_wins}")
    print(f"Draws: {draw}")


start_game()