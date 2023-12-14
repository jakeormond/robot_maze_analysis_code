import os

def get_home_dir():
    # determine operating system
    if os.name == 'nt':
        home_dir = 'D:/analysis' # WINDOWS
    elif os.name == 'posix': # Linux or Mac OS
        home_dir = "/media/jake/LaCie" # Linux/Ubuntu
    return home_dir

def get_data_dir(animal, session):
    home_dir = get_home_dir()
    data_dir = os.path.join(home_dir, animal, session)
    return data_dir


if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)
    bonsai_dir = os.path.join(data_dir, 'video_csv_files')
    print(bonsai_dir)