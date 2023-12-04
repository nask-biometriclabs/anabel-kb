from model import load_model

if __name__ == '__main__':

    checkpoints = ["/home/weronika/Projekty/anabel-ka/data/checkpoints/AnabelKA_mobile_2023-10-29-215204-nfix.pkl",
                   "/home/weronika/Projekty/anabel-ka/data/checkpoints/AnabelKA_desktop_2023-10-29-224950-nfix.pkl"]

    for ch in checkpoints:
        model = load_model(ch)