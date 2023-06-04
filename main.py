from interface import App

if __name__ == "__main__":
    app = App()
    app.mainloop()


# pyinstaller --noconfirm --onedir --windowed --add-data "F:/programming/projects/bina_project-bina-v.1.1/.venv/lib/site-packages/customtkinter;customtkinter/"  main.py
# or
# pyinstaller main.spec
