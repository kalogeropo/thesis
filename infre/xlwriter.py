import pandas as pd
import os.path

class ExcelWriter:
    def __init__(self, file_path):
        self.file_path = file_path

    def check_file_exists(self):
        return os.path.isfile(self.file_path)

    def create_file(self, sheet_name, dataframe):
        dataframe.to_excel(self.file_path, sheet_name=sheet_name, index=False)

    def append_to_sheet(self, sheet_name, dataframe):
        with pd.ExcelWriter(self.file_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)

    def write_to_excel(self, sheet_name, dataframe):
        try:
            if not self.check_file_exists():
                self.create_file(sheet_name, dataframe)
                print(f"Created file '{self.file_path}' and stored the dataframe in sheet '{sheet_name}'.")
            else:
                self.append_to_sheet(sheet_name, dataframe)
                print(f"Appended the dataframe to sheet '{sheet_name}' in file '{self.file_path}'.")
        except:
            print(f"Failed to write dataframe to sheet '{sheet_name}' in file '{self.file_path}'.")
