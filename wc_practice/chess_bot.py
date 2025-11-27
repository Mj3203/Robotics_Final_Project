from stockfish import Stockfish
import chess

# Load engine
stockfish = Stockfish(path=r"C:\Users\Micha\Downloads\stockfish-windows-x86-64-avx2.exe")

# Create the board
board = chess.Board()

# ================================
# 🔥 ROBOT (STOCKFISH) PLAYS FIRST
# ================================
stockfish.set_fen_position(board.fen())
engine_move = stockfish.get_best_move()

print("\nRobot plays first:", engine_move)

board.push_uci(engine_move)    # Apply robot's move on the board

# ================================
# 🔥 NOW START NORMAL GAME LOOP
# ================================
while not board.is_game_over():
    print("\nCurrent position:")
    print(board)

    # === Human move ===
    user_move = input("\nYour move (UCI): ")

    if user_move not in [m.uci() for m in board.legal_moves]:
        print("Illegal move, try again.")
        continue

    board.push_uci(user_move)

    # === Update Stockfish position ===
    stockfish.set_fen_position(board.fen())

    # === Stockfish calculates new move ===
    engine_move = stockfish.get_best_move()

    print("Robot plays:", engine_move)

    board.push_uci(engine_move)



##saving this for now
    while True:
        # --- ROBOT TURN ---
        engine_move = stockfish.get_best_move()
        robot.execute(engine_move)

        # Save board after robot moves
        board_before = get_board_state()  # YOLO → dict
        print("Saved board_before")

        # --- HUMAN TURN ---
        wait_for_button_press()  # block until human presses button

        # Save board after human moves
        board_after = get_board_state()  # YOLO → dict
        print("Saved board_after")

        # Determine human move
        human_move = detect_move(board_before, board_after)
        print("Human played:", human_move)

        # Give human move to stockfish
        stockfish.set_fen_position(apply_move_to_fen(human_move))

        # Loop repeats
