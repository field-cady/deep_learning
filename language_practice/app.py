import streamlit as st
import language_tutor


st.write("Welcome! Click a sentence to see pronunciation and translation.")

# Only get sentences once until the app is rerun
try:
    lt = st.session_state.lt
    sentences = st.session_state.sentences
except:
    with st.spinner("Generating Sentences..."):
        lt = language_tutor.LanguageTutor()
        sentences = lt.get_sentences(n=10)
        print('target words:', lt.state['target_words'])
        print(sentences)
        st.session_state.lt = lt
        st.session_state.sentences = sentences

if "clicked" not in st.session_state:
    st.session_state.clicked = {s: False for tw, s, pron, trans in sentences}

for i, (tw, s, pron, trans) in enumerate(sentences):
    item = s
    col1, col2, col3 = st.columns([4, 1, 1])

    # Main text button
    with col1:
        if st.button(s, key=f"text_{item}"):
            st.session_state.clicked[item] = True

    with col2:
        st.toggle('Failed', key=item)

    # 4. Show text under the button if clicked
    if st.session_state.clicked[item]:
        st.write(pron)
        st.write(trans)

errors = []
for i, (tw, s, pron, trans) in enumerate(sentences):
    errors.append(i)

col1, col2, col3 = st.columns([4, 1, 1])
with col1:
    if st.button('Submit', key=f"submit"):
        new_words = lt.update(sentences, errors)
        lt.save()
        if new_words:
            for nw in new_words:
                st.write(nw)
            time.sleep(5)
        st.session_state.clear()
        st.rerun()

