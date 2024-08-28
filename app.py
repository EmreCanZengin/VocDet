import os
from modules.enrollment import *
from sklearn.svm import SVC
from modules.record import *
import pickle

PROGRAM_VERSION = "0.0.1"


def print_banner():
    print("Welcome to Project VocDet")
    print("VocDet version: ", PROGRAM_VERSION)
    print("Loading The Data...")

def print_info():
    print("Information about the program...")

def print_commands(ls):
    print("Available commands:")
    for i, cmd in enumerate(ls):
        print(f"{i+1}. {cmd}")

def main_menu():
    ls = ["back","help", "quit", "exit", "enrollment", "verify", "identify", "clear", "ls"]
    while True:
        print("""
        lalalalalalllalallalaldsjkfalllllllll
        dsk;lfajjjjjjjjjjj ...l.
        """)
        ans = input(">>> ").strip().lower()

        if ans == "back":
            return False
        if ans in ["quit", "exit"]:
            return True  # Signal to exit the program
        elif ans == "help":
            print_info()
        elif ans == "enrollment":
            if enrollment_menu():
                return True
        elif ans == "verify":
            if verify_menu():
                return True
        elif ans == "identify":
            if identify_menu():
                return True
        elif ans == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')
        elif ans == "ls":
            print_commands(ls)
        else:
            print("Unknown command. Please try again.")

def enrollment_menu():
    ls = ["back", "help", "quit", "exit", "new", "delete", "clear", "ls"]
    while True:
        ans = input("(enrollment) >>> ").strip().lower()
        if ans == "back":
            return False

        elif ans == "help":
            print_info()

        elif ans in ["quit", "exit"]:
            return True

        elif ans == "ls":
            print_commands(ls)

        elif ans == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')

        elif ans.split()[0] == "new":
            for enroll_name in ans.split()[1:]:
                if enroll_name not in current_enrollments.keys():
                    current_enrollments[enroll_name] = Enrollment(enroll_name)
                    modified_enrollments.add(enroll_name)
                else: 
                    print(f"Enrollment '{enroll_name}' already exists.")

        elif ans in current_enrollments.keys():
            if enrollment_opt_menu(ans):
                return True

        elif ans == "ls -e":
            print("Current enrollements: ")
            print(list(current_enrollments.keys()))

        elif ans.split()[0] == "delete":
            for enroll_name in ans.split()[1:]:
                del current_enrollments[enroll_name]
                modified_enrollments.remove(enroll_name)

        elif ans.split()[0] == "clean":
            for enroll_name in ans.split()[1:]:
                current_enrollments[enroll_name].cleanEnrollment()
        else:
            print("Unknown Command. Please try again.")

def enrollment_opt_menu(enroll_name:str):
    ls = ["back","help", "quit", "exit", "add", "delete", "clear", "ls"]
    while True:
        ans = input(f"(enrollment/{enroll_name}) >>> ").strip().lower()
        if ans == "back":
            ans1 = input("You modified this enrollment do you want to update its model? (y/n): ")
            if ans1 == "y":
                current_enrollments[enroll_name].updateModel()
                modified_enrollments.remove(enroll_name)
            return False
        elif ans == "help":
            print_info()
        elif ans in ["quit", "exit"]:
            return True
        elif ans == "add":
            if enrollment_speaker_add_menu(enroll_name):
                return True
        elif ans == "delete":
            if enrollment_delete_menu(enroll_name):
                return True
        elif ans == "ls":
            print_commands(ls)
        elif ans == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')
        elif ans == "update":
            current_enrollments[enroll_name].updateModel()
        else:
            print("Unknown Command. Please try again.")

def enrollment_speaker_add_menu(enroll_name):
    ls = ["back","help", "quit", "exit", "new", "clear", "ls"]
    while True:
        ans = input(f"(enrollment/{enroll_name}/add) >>> ").strip().lower()
        if ans == "back":
            return False
        elif ans == "help":
            print_info()
        elif ans in ["quit", "exit"]:
            return True
        elif ans.split()[0] == "new":
            for speaker_name in ans.split()[1:]:
                current_enrollments[enroll_name].addSpeaker(speaker_name)

        elif ans == "ls -s":
            print(f"{enroll_name}'s speakers: ")
            print(list(current_enrollments[enroll_name].known_speakers))

        elif ans in current_enrollments[enroll_name].known_speakers:
            if add_speech(enroll_name, ans):
                return True
            
        elif ans == "ls":
            print_commands(ls)

        elif ans == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')
        else:
            print("Unknown Command. Please try again.")

def enrollment_delete_menu(enroll_name):
    ls = ["back","help", "quit", "exit", "speaker", "speech","clear"]
    while True:
        ans = input(f"(enrollment/{enroll_name}/delete) >>> ").strip().lower()
        if ans == "back":
            return False

        elif ans == "help":
            print_info()
        elif ans in ["quit", "exit"]:
            return True
        elif ans in current_enrollments[enroll_name].known_speakers.keys():
            if delete_speaker_speech(enroll_name, ans):
                return True
        elif ans == "ls":
            print_commands(ls)

        elif ans == "ls -s":
            print(f"{enroll_name}'s speakers: ")
            print(list(current_enrollments[enroll_name].known_speakers))

        elif ans == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')
        else:
            print("Unknown Command. Please try again.")

def verify_menu():
    ls = ["back", "quit", "exit", "clear", "ls"]
    while True:
        ans = input("(verify) >>> ").strip().lower()
        if ans == "back":
            return False
        elif ans == "help":
            print_info()
        elif ans in ["quit", "exit"]:
            return True
        elif ans == "ls":
            print_commands(ls)
        
        elif ans == "ls -e":
            print("Current enrollements: ")
            print(list(current_enrollments.keys()))

        elif ans == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')

        elif ans in current_enrollments.keys():
            if ans not in modified_enrollments:
                claimed_name = input("Please Enter the name you want to verify: ")
                handle_verification(ans, claimed_name)
            else:
                print("This enrollment probably doesn't have any data in it. Please add data.")
        else:
            print("Verification here")
            print("If you want to go back use `back` command, quit use `quit` or `exit`.")

def identify_menu():
    ls = ["back","help", "quit", "exit", "clear", "ls"]
    while True:
        ans = input("(identify) >>> ").strip().lower()
        if ans == "back":
            return False
        elif ans == "help":
            print_info()
        elif ans in ["quit", "exit"]:
            return True
        elif ans == "ls":
            print_commands(ls)
        elif ans == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')

        elif ans == "ls -e":
            print("Current enrollements: ")
            print(list(current_enrollments.keys()))

        elif ans in current_enrollments.keys():
            if ans not in modified_enrollments:
                handle_identification(ans)
            else:
                print("This enrollment probably doesn't have any data in it. Please add data.")
        else:
            print("identification here")
            print("If you want to go back use `back` command, quit use `quit` or `exit`.")

def add_speech(enroll_name, speaker):
    ls = ["back","help", "quit", "exit", "clear", "ls"]
    while True:
        ans = input(f"(enrollment/{enroll_name}/add/{speaker}) >>> ").strip().lower()
        if ans == "back":
            return False
        elif ans == "help":
            print_info()
        elif ans in ["quit", "exit"]:
            return True
        elif ans == "ls":
            print_commands(ls)
        elif ans == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')
        elif ans == "ls -sp":
            print(f"{speaker}'s speeches: ", current_enrollments[enroll_name].known_speakers[speaker].speeches)
        elif ans == "record":
            modified_enrollments.add(enroll_name)
            speaker_id = current_enrollments[enroll_name].known_speakers[speaker].id
            filename = Path(f"audio/{enroll_name}/{speaker}_{speaker_id}/{speaker}_{current_enrollments[enroll_name].known_speakers[speaker].n_speech}.wav")
            current_enrollments[enroll_name].addSpeech(speaker, filename)
            record_audio(filename)
        else:
            print("Adding Speech here")
            print("If you want to go back use `back` command, quit use `quit` or `exit`.")

def delete_speaker_speech(enroll_name, speaker):
    ls = ["back", "quit", "exit", "clear", "ls"]
    print("You can delete speaker with `self` or you can delete a speech from this speaker")

    while True:
        ans = input(f"(enrollment/{enroll_name}/delete/{speaker}) >>> ").strip().lower()

        if ans == "back":
            return False

        elif ans == "help":
            print_info()

        elif ans in ["quit", "exit"]:
            return True

        elif ans == "ls":
            print_commands(ls)

        elif ans == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')
        
        elif ans == "self":
            modified_enrollments.add(enroll_name)
            if current_enrollments[enroll_name].known_speakers[speaker].n_speech > 0:
                ans1 = input("This speaker has speeches. Are you sure you want to delete the speaker? (y/n) ").lower()
                if ans1 == "y":
                    del current_enrollments[enroll_name].known_speakers[speaker]
                    print(f"Speaker '{speaker}' deleted.")
                else:
                    print(f"Speaker '{speaker}' couldn't deleted.")
            else:
                del current_enrollments[enroll_name].known_speakers[speaker]
                print(f"Speaker '{speaker}' deleted.")
                break

        elif ans == "ls -sp":
            print(f"{speaker}'s speeches: ", current_enrollments[enroll_name].known_speakers[speaker].speeches)

        elif ans == "speech":
            modified_enrollments.add(enroll_name)
            current_enrollments[enroll_name].deleteSpeechs(current_enrollments[enroll_name].known_speakers[speaker])
        else:
            print("Deleting Speaker or Speech here")
            print("If you want to go back use `back` command, quit use `quit` or `exit`.")

def handle_verification(enroll_name, claimed_speaker_name):
        filename = Path(f"audio/unknown/unknown.wav")
        record_audio(filename)
        X = current_enrollments[enroll_name].preprocessing.transform(np.array([filename]))
        id_pred = current_enrollments[enroll_name].model.predict(X)
        if id_pred == -1:
            print("Verification failed")
        else:
            try:
                id_claimed = current_enrollments[enroll_name].known_speakers[claimed_speaker_name].id
            
                if id_claimed == id_pred:
                    print("Verification is successfull")

                else:
                    print("Verification is failed")
            except:
                print("Claimed Speaker name is unknown")

def handle_identification(enroll_name):
        filename = Path(f"audio/unknown/unknown.wav")
        record_audio(filename)
        X = current_enrollments[enroll_name].preprocessing.transform(np.array([filename]))
        id = current_enrollments[enroll_name].model.predict(X)

        if id == -1:
            print("Speaker is unknown")
        else: 
            print("Identification is successfull")
            id_name = current_enrollments[enroll_name].findSpeakerNameWithId(id) 
            print(f"Identified person: {id_name}")


def loadEnrollments():
    try:
        with open('enrollments.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return (dict(), set())

def saveEnrollments():
    global modified_enrollments
    if len(modified_enrollments) > 0:
        ans = input("""
[WARNING] You didn't update the models of the modified enrollments.
If you have time, the program can update it.
If you have to exit, the program will store modified enrollments too.
Update: u, save and exit: (anything except u): """)

        if ans == "u":
            for enroll_name in modified_enrollments:
                current_enrollments[enroll_name].updateModel()
            modified_enrollments = set()

    with open('enrollments.pkl', 'wb') as file:
        pickle.dump((current_enrollments, modified_enrollments), file)

if __name__ == "__main__":
    print_banner()
    current_enrollments, modified_enrollments = loadEnrollments()

    if len(modified_enrollments) > 0:
        print("Modified enrollments detected. Updating models...")
        for enroll_name in modified_enrollments:
            current_enrollments[enroll_name].updateModel()
        modified_enrollments.update(set())
    modified_enrollments= set()

    model = SVC(C= 5, probability = True)
    quit_program = main_menu()
    if quit_program:
        print("\nSaving the data...")
        print("Goodbye")
    saveEnrollments()