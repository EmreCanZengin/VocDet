from modules.enrollment import *
import pickle

def enrollmentFolderToEnrollments(enrollment_folder: Path, enrollement_file= Path("enrollments.pkl")):
    try:
        enrolls, modified = pickle.load(enrollement_file)
    except:
        enrolls = dict()
        modified = set()

    for enrollment in enrollment_folder.iterdir():
        str_enrollment = str(enrollment.relative_to(enrollment_folder)).lower()
        if not enrolls.get(str_enrollment, 0):
            enrolls[str_enrollment] = Enrollment(str_enrollment)
        for speaker in enrollment.iterdir():
            enrolls[str_enrollment].addSpeaker(str(speaker.relative_to(enrollment)).lower())
            for file in speaker.iterdir():
                enrolls[str_enrollment].addSpeech(str(speaker.relative_to(enrollment)).lower(), file.absolute())
        modified.add(str_enrollment)
    
    with open('enrollments.pkl', 'wb') as file:
        pickle.dump((enrolls, modified), file)


if __name__ == "__main__":
    enrollmentFolderToEnrollments(Path("./enrolls"))