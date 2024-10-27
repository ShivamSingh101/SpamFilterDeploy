import streamlit as st
from prediction import predict

def main():
    st.set_page_config(
        page_title="Spam Filter",
        page_icon=":mail",
    )
    st.title("Spam email classification")
    st.write("Upload the mail to check its spam or not.")
    input=st.text_area(":blue[Enter text:]",height=150)
    if st.button("Check Mail"):
        st.write(predict(input))

if __name__ == "__main__":
    main()