import streamlit as st
import random
from openai import OpenAI

st.set_page_config(page_title="Tic-Tac-Toe")

# -----------------------------
# CSS styling for nicer board
# -----------------------------
st.markdown("""
<style>
/* Board buttons */
div.stButton > button {
    border: 3px solid #333;         /* darker border */
    border-radius: 8px;             /* slightly rounded */
    background-color: #f8f8f8;      /* subtle gray */
    color: #111;
    width: 80px;                     /* fixed square size */
    height: 80px;
    font-size: 32px;
    font-weight: bold;
    margin: 2px;
    transition: background-color 0.2s, transform 0.1s;
}
div.stButton > button:hover {
    background-color: #e0e0e0;      /* hover highlight */
    transform: scale(1.05);
}

/* Center board and messages */
.css-1kyxreq {  /* Streamlit main container */
    display: flex;
    flex-direction: column;
    align-items: center;
}

/* Message style */
div[data-testid="stMarkdownContainer"] p {
    font-size: 20px;
    font-weight: bold;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 10px;
}

/* Restart button style */
div.stButton > button:active {
    background-color: #d0d0d0;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# OpenAI setup (unchanged)
# -----------------------------
api_key = open('openai_api_key.txt', 'r').read()
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

# -----------------------------
# Initialize game state
# -----------------------------
if "board" not in st.session_state:
    st.session_state.board = [""] * 9
    st.session_state.game_over = False
    st.session_state.user_symbol = random.choice(["X", "O"])
    st.session_state.computer_symbol = "O" if st.session_state.user_symbol == "X" else "X"
    st.session_state.turn = random.choice(["user", "computer"])
    st.session_state.message = f"You are {st.session_state.user_symbol}. {st.session_state.turn.capitalize()} goes first!"

# -----------------------------
# Helper functions (unchanged)
# -----------------------------
def check_winner(board):
    lines = [
        [0,1,2], [3,4,5], [6,7,8],
        [0,3,6], [1,4,7], [2,5,8],
        [0,4,8], [2,4,6]
    ]
    for line in lines:
        a,b,c = line
        if board[a] == board[b] == board[c] != "":
            return board[a]
    if "" not in board:
        return "Tie"
    return None

def computer_move():
    with st.spinner("Computer is Thinking..."):
        new_board = get_new_board(st.session_state.board, st.session_state.computer_symbol)
    st.session_state.board = new_board

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
# Render board as 3x3 grid (centered)
# -----------------------------
for row in range(3):
    cols = st.columns(3)
    for col in range(3):
        idx = row * 3 + col
        disab = bool(st.session_state.board[idx])
        if cols[col].button(st.session_state.board[idx] or " ", key=f"display_{idx}", disabled=disab):
            st.session_state.board[idx] = st.session_state.user_symbol
            st.session_state.turn = "computer"
            st.rerun()

# -----------------------------
# Show message nicely
# -----------------------------
st.write(st.session_state.message)

# -----------------------------
# Restart button
# -----------------------------
import streamlit as st


if st.button("New Game"):
    st.write("Restart clicked!")
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


