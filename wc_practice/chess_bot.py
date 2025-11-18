from stockfish import Stockfish
import chess

# load engine
stockfish = Stockfish(path=r"C:\Users\Micha\Downloads\stockfish-windows-x86-64-avx2.exe")

# create chess board
board = chess.Board()

while not board.is_game_over():
    print("\nCurrent position:")
    print(board)

    # === 1. Get user's UCI move ===
    # Example: "e2e4"
    user_move = input("\nEnter your move (UCI): ")

    # Validate move
    if user_move not in [m.uci() for m in board.legal_moves]:
        print("Illegal move, try again.")
        continue

    # === 2. Apply user's move ===
    board.push_uci(user_move)

    # === 3. Give new position to Stockfish ===
    stockfish.set_fen_position(board.fen())

    # === 4. Stockfish calculates best move ===
    engine_move = stockfish.get_best_move()

    print("Stockfish plays:", engine_move)

    # === 5. Apply engine move to board ===
    board.push_uci(engine_move)