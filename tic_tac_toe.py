import streamlit as st
import random
from openai import OpenAI


st.markdown("""
<style>
div.stButton > button {
    border: 2px solid black;   /* dark border */
    border-radius: 5px;        /* optional rounded corners */
    background-color: #f0f0f0; /* button background */
    color: black;              /* text color */
    height: 60px;              /* optional fixed height */
    font-size: 24px;           /* larger text */
    font-weight: bold;         /* bold text */
}
</style>
""", unsafe_allow_html=True)


api_key=open('openai_api_key.txt', 'r').read()
client = OpenAI(api_key=api_key)


PROMPT_TEMPLATE = """
You are playing tic-tac-toe.
The state of the board is a list of strings,
where each string is the empty string "",
"X" or "O".
The empty string means you can play there.

I will tell you the state of the board
and whether you are X or O.

You decide on the best move to make,
and output the state of the board after that move.


For example, if the current state is:
['', '', '', '', '', '', '', '', '']
and you are X, you might make it
['X', '', '', '', '', '', '', '', '']

You are {chr} and the board is:
{board}

After your move the board is:
"""
def get_new_board(board, chr, model="gpt-5-mini"):
    prompt = PROMPT_TEMPLATE.format(**locals())
    response = client.responses.create(
        model = model,
        input = prompt
    )
    return eval(response.output_text)

st.set_page_config(page_title="Tic-Tac-Toe")


# -----------------------------
# Initialize game state
# -----------------------------
if "board" not in st.session_state:
    st.session_state.board = [""] * 9  # 9 empty cells
    st.session_state.game_over = False
    # Randomly assign X or O to user and computer
    st.session_state.user_symbol = random.choice(["X", "O"])
    st.session_state.computer_symbol = "O" if st.session_state.user_symbol == "X" else "X"
    # Randomly choose who goes first
    st.session_state.turn = random.choice(["user", "computer"])
    st.session_state.message = f"You are {st.session_state.user_symbol}. {st.session_state.turn.capitalize()} goes first!"

# -----------------------------
# Helper functions
# -----------------------------
def check_winner(board):
    lines = [
        [0,1,2], [3,4,5], [6,7,8],  # rows
        [0,3,6], [1,4,7], [2,5,8],  # columns
        [0,4,8], [2,4,6]            # diagonals
    ]
    for line in lines:
        a, b, c = line
        if board[a] == board[b] == board[c] != "":
            return board[a]
    if "" not in board:
        return "Tie"
    return None

def computer_move():
    new_board = get_new_board(st.session_state.board, st.session_state.computer_symbol)
    st.session_state.board = new_board
    #empty_indices = [i for i, x in enumerate(st.session_state.board) if x == ""]
    #if empty_indices:
    #    move = random.choice(empty_indices)
    #    st.session_state.board[move] = st.session_state.computer_symbol

# -----------------------------
# Computer move
# -----------------------------
if not st.session_state.game_over and st.session_state.turn == "computer":
    computer_move()
    st.session_state.turn = "user"

# -----------------------------
# Check for winner
# -----------------------------
winner = check_winner(st.session_state.board)
if winner:
    if winner == "Tie":
        st.session_state.message = "It's a tie!"
    else:
        st.session_state.message = f"{winner} wins!"
    st.session_state.game_over = True

# -----------------------------
# Render board as 3x3 grid
# -----------------------------
for row in range(3):
    cols = st.columns(3)
    for col in range(3):
        idx = row * 3 + col
        disab = bool(st.session_state.board[idx]) # disable if square already filled
        if cols[col].button(st.session_state.board[idx] or " ", key=f"display_{idx}", disabled=disab):
            st.session_state.board[idx] = st.session_state.user_symbol
            st.session_state.turn = "computer"
            st.rerun()

# -----------------------------
# Show message
# -----------------------------
st.write(st.session_state.message)

# -----------------------------
# Restart button
# -----------------------------
if st.button("Restart"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
