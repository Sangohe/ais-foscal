import os
import ntpath
from glob import glob
from typing import Union, List

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

from .config import get_path_of_directory_with_id

def path_leaf(path: str):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def upload_file_to_dir_in_drive(drive, filepath: str, parent_id: str):
    file_dict = {
        'name': path_leaf(filepath),
        'title': path_leaf(filepath),
        'parents': [{'id': parent_id}]
    }
    gfile = drive.CreateFile(file_dict)
    gfile.SetContentFile(filepath)
    gfile.Upload()

def get_subdirs_in_dir(drive, dir_id: str):
    queries = [
        "mimeType='application/vnd.google-apps.folder'",
        "trashed=false",
        f"'{dir_id}' in parents"
    ]
    return drive.ListFile({
        'q': ' and '.join(queries)
    }).GetList()

def create_subdir_in_drive(drive, dirname: str, parent_dir_id: str):
    """Creates a subdir inside parent directory using an already 
    auth instance of drive."""
    file_metadata = {
        'name': dirname,
        'title': dirname,
        'parents': [{'id': parent_dir_id}],
        'mimeType': 'application/vnd.google-apps.folder'
    }
    
    subdirs_in_dir = get_subdirs_in_dir(drive, parent_dir_id)
    subdirs_titles = [d['title'] for d in subdirs_in_dir]
    
    if not dirname in subdirs_titles:
        print(f'Creating {dirname}...')
        drive_dir = drive.CreateFile(file_metadata)
        drive_dir.Upload()
        return drive_dir
    else:
        print(f'{dirname} already exists.')
        return subdirs_in_dir[subdirs_titles.index(dirname)]

def get_authenticated_drive_instance():
    
    gauth = GoogleAuth()
    auth_url = gauth.GetAuthUrl()
    print(auth_url)
    token = input('Paste the taken you got for authorization: ')
    gauth.Auth(token)
    
    # Create GDrive instance with the authenticated GoogleAuth instance.
    drive = GoogleDrive(gauth)
    return drive
    
def push_experiment_dir_to_drive(
    drive,
    exp_id: Union[int, str], 
    dirs_to_upload: List[str] = ['config', 'logs'],
    dest_drive_dir_id: str = '1x6_OPwLD2USPb4Goe8-BqbPzFYcWdRGp',
    results_dir: str = 'results'
):

    drive_dirs = {}
    exp_dir = get_path_of_directory_with_id(exp_id, results_dir=results_dir)
    dirs_to_upload = [os.path.join(exp_dir, d) for d in dirs_to_upload]
    
    for root, dirs, files in os.walk(exp_dir):
        
        # Check if the directory I am iterating is a subdirectory 
        # of the allowed directories.
        root_in_dirs_to_upload = any(
            root.startswith(dir_to_upload) 
            for dir_to_upload in dirs_to_upload
        )

        if root == exp_dir or root_in_dirs_to_upload:
            
            # Create dir.
            dirname = os.path.dirname(root)
            drive_dir_id = drive_dirs[dirname]['id'] if drive_dirs else dest_drive_dir_id
            root_drive_dir = create_subdir_in_drive(
                drive, path_leaf(root), drive_dir_id)
            drive_dirs[root] = {
                'id': root_drive_dir['id'], 
                'title': root_drive_dir['title'], 
                'name': root_drive_dir['name']
            }
            
            # Upload files.
            for file in files:
                filepath = os.path.join(root, file)
                print(f'\tUploading {filepath} to {root_drive_dir["title"]}')
                upload_file_to_dir_in_drive(drive, filepath, root_drive_dir['id'])