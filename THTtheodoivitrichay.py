import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import qrcode
import io
import json  # Import the json library
import atexit # Import the atexit module

# --- Data Storage (File-Based) ---
DATA_FILE = "fire_data.json"
users = {} # Keep users in memory for simplicity in this example

# Load existing citizen reports from file if it exists
try:
    with open(DATA_FILE, 'r') as f:
        citizen_reports = json.load(f)
except FileNotFoundError:
    citizen_reports = {}
except json.JSONDecodeError:
    citizen_reports = {}

fire_announced = False

class FireTrackerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fire Location Tracker (Prototype)")
        self.geometry("500x600")

        self.current_user_id = None
        self.current_user_type = None

        self._container = tk.Frame(self)
        self._container.pack(side="top", fill="both", expand=True)
        self._container.grid_rowconfigure(0, weight=1)
        self._container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartScreen, SignUpScreen, CitizenWidgetScreen, CitizenActiveScreen, GovernmentScreen, GovernmentBuildingDetailScreen):
            page_name = F.__name__
            frame = F(parent=self._container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartScreen")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        if page_name == "GovernmentScreen":
            frame.refresh_display()
        elif page_name == "CitizenActiveScreen" and self.current_user_id:
            frame.generate_and_display_qr()
        frame.tkraise()

    def get_frame(self, page_name):
        return self.frames[page_name]

    def set_current_user(self, user_id, user_type):
        self.current_user_id = user_id
        self.current_user_type = user_type
        print(f"User set: ID={user_id}, Type={user_type}")

    def clear_current_user(self):
        self.current_user_id = None
        self.current_user_type = None

    def register_user(self, user_id, name, user_type):
        if not user_id or not name:
            messagebox.showerror("Error", "ID and Name cannot be empty.")
            return False
        if user_id in users:
            messagebox.showerror("Error", f"User ID '{user_id}' already exists.")
            return False
        users[user_id] = {"name": name, "type": user_type}
        if user_type == 'citizen':
            citizen_reports[user_id] = {"status": "OFF", "people": 0, "building": "", "floor": "", "room": ""}
            self.save_citizen_reports() # Save on registration for consistency
        print(f"Registered: {users[user_id]}")
        return True

    def update_citizen_report(self, user_id, status, people, building, floor, room):
        global fire_announced, citizen_reports

        # Load the latest citizen reports from the file before updating
        try:
            with open(DATA_FILE, 'r') as f:
                citizen_reports = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            citizen_reports = {}

        if user_id not in users or users[user_id]['type'] != 'citizen':
            messagebox.showerror("Error", "Invalid citizen user.")
            return False

        try:
            people_count = int(people) if status == "ON" else 0
            if people_count < 0: raise ValueError()
        except ValueError:
            messagebox.showerror("Error", "Number of people must be a non-negative integer.")
            return False

        if status == "ON" and (not building or not floor):
            messagebox.showerror("Error", "Building and Floor/Room are required when status is ON.")
            return False

        citizen_reports[user_id] = {
            "status": status,
            "people": people_count,
            "building": building.strip(),
            "floor": floor.strip(),
            "room": room.strip()
        }
        print(f"Updated report for {user_id}: {citizen_reports[user_id]}")
        self.save_citizen_reports() # Save after updating report
        self.recalculate_fire_data()
        return True

    def set_citizen_off(self, user_id):
        if user_id in citizen_reports:
            citizen_reports[user_id]['status'] = 'OFF'
            citizen_reports[user_id]['people'] = 0
            self.save_citizen_reports() # Save after setting OFF
            print(f"Set citizen {user_id} to OFF via QR simulation.")
            self.recalculate_fire_data()
            return True
        return False

    def recalculate_fire_data(self):
        global fire_announced
        any_on = any(report['status'] == 'ON' for report in citizen_reports.values())
        fire_announced = any_on
        print(f"Recalculating: Fire Announced = {fire_announced}")
        if self.frames.get("GovernmentScreen") and self.frames["GovernmentScreen"].winfo_ismapped():
            self.frames["GovernmentScreen"].refresh_display()

    def get_aggregated_fire_data(self):
        aggregated_data = {}
        for report in citizen_reports.values():
            if report['status'] == 'ON':
                building = report['building']
                floor = report['floor']
                room = report['room'] if report['room'] else "Unknown Room"
                people = report['people']

                if building not in aggregated_data:
                    aggregated_data[building] = {'total': 0, 'floors': {}}
                aggregated_data[building]['total'] += people

                if floor not in aggregated_data[building]['floors']:
                    aggregated_data[building]['floors'][floor] = {'total': 0, 'rooms': {}}
                aggregated_data[building]['floors'][floor]['total'] += people

                if room not in aggregated_data[building]['floors'][floor]['rooms']:
                    aggregated_data[building]['floors'][floor]['rooms'][room] = 0
                aggregated_data[building]['floors'][floor]['rooms'][room] += people
        return aggregated_data

    def save_citizen_reports(self):
        try:
            # Load existing reports from file
            try:
                with open(DATA_FILE, 'r') as f:
                    existing_reports = json.load(f)
            except FileNotFoundError:
                existing_reports = {}
            except json.JSONDecodeError:
                existing_reports = {}

            # Merge the current instance's report with the existing reports
            # We'll iterate through the current citizen_reports and update/add to existing_reports
            for user_id, report in citizen_reports.items():
                existing_reports[user_id] = report

            # Save the merged reports back to the file
            with open(DATA_FILE, 'w') as f:
                json.dump(existing_reports, f)
            print(f"Citizen reports saved to {DATA_FILE}. Content: {existing_reports}")
        except Exception as e:
            print(f"Error saving citizen reports: {e}")

    def load_citizen_reports(self):
        global citizen_reports
        try:
            print(f"Attempting to open file: {DATA_FILE}")
            with open(DATA_FILE, 'r') as f:
                print(f"File {DATA_FILE} opened successfully.")
                file_content = f.read()
                print(f"Content of {DATA_FILE}: '{file_content}'")
                if file_content:
                    citizen_reports = json.loads(file_content)
                    print(f"JSON loaded successfully. citizen_reports: {citizen_reports}")
                else:
                    citizen_reports = {}
                    print(f"File {DATA_FILE} is empty. Initializing citizen_reports to: {citizen_reports}")
        except FileNotFoundError:
            citizen_reports = {}
            print(f"{DATA_FILE} not found. Initializing citizen_reports to: {citizen_reports}")
        except json.JSONDecodeError as e:
            citizen_reports = {}
            print(f"Error decoding JSON from {DATA_FILE}: {e}")
            print(f"Initializing citizen_reports to: {citizen_reports}")
        except Exception as e:
            citizen_reports = {}
            print(f"An unexpected error occurred during loading: {e}")
            print(f"Initializing citizen_reports to: {citizen_reports}")


class StartScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="Select User Type", font=("Arial", 18))
        label.pack(pady=20, padx=10)

        citizen_btn = tk.Button(self, text="Citizen", command=lambda: self.go_to_signup('citizen'))
        citizen_btn.pack(pady=10, fill=tk.X, padx=50)

        gov_btn = tk.Button(self, text="Government", command=lambda: self.go_to_signup('government'))
        gov_btn.pack(pady=10, fill=tk.X, padx=50)

    def go_to_signup(self, user_type):
        signup_frame = self.controller.get_frame("SignUpScreen")
        signup_frame.set_user_type(user_type)
        self.controller.show_frame("SignUpScreen")


class SignUpScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.user_type_to_register = None

        tk.Label(self, text="Register / Sign In", font=("Arial", 16)).pack(pady=10)

        tk.Label(self, text="ID Number:").pack(pady=5)
        self.id_entry = tk.Entry(self)
        self.id_entry.pack(pady=5, padx=20, fill=tk.X)

        tk.Label(self, text="Name:").pack(pady=5)
        self.name_entry = tk.Entry(self)
        self.name_entry.pack(pady=5, padx=20, fill=tk.X)

        register_btn = tk.Button(self, text="Register / Continue", command=self.process_registration)
        register_btn.pack(pady=20)

        back_btn = tk.Button(self, text="Back", command=lambda: controller.show_frame("StartScreen"))
        back_btn.pack(pady=5)

    def set_user_type(self, user_type):
        self.user_type_to_register = user_type
        tk.Label(self, text=f"Register as {user_type.capitalize()}", font=("Arial", 16)).pack(pady=10)

    def process_registration(self):
        user_id = self.id_entry.get()
        name = self.name_entry.get()

        if not self.user_type_to_register:
            messagebox.showerror("Error", "User type not selected.")
            return

        if user_id in users:
            if users[user_id]['type'] == self.user_type_to_register:
                messagebox.showinfo("Login", f"Welcome back, {users[user_id]['name']}!")
                self.controller.set_current_user(user_id, self.user_type_to_register)
                if self.user_type_to_register == 'citizen':
                    self.controller.show_frame("CitizenWidgetScreen")
                else:
                    self.controller.show_frame("GovernmentScreen")
            else:
                messagebox.showerror("Error", f"User ID '{user_id}' exists but is not a {self.user_type_to_register}.")
        else:
            if self.controller.register_user(user_id, name, self.user_type_to_register):
                messagebox.showinfo("Success", "Registration successful!")
                self.controller.set_current_user(user_id, self.user_type_to_register)
                if self.user_type_to_register == 'citizen':
                    self.controller.show_frame("CitizenWidgetScreen")
                else:
                    self.controller.show_frame("GovernmentScreen")

        self.id_entry.delete(0, tk.END)
        self.name_entry.delete(0, tk.END)


class CitizenWidgetScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Label(self, text="Citizen Emergency Report", font=("Arial", 16)).pack(pady=10)

        self.status_var = tk.StringVar(value="OFF")
        status_frame = tk.Frame(self)
        tk.Radiobutton(status_frame, text="Tracking ON", variable=self.status_var, value="ON", command=self.toggle_inputs).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(status_frame, text="Tracking OFF", variable=self.status_var, value="OFF", command=self.toggle_inputs).pack(side=tk.LEFT, padx=10)
        status_frame.pack(pady=10)

        self.input_frame = tk.Frame(self)

        tk.Label(self.input_frame, text="Number of People (including yourself):").pack(anchor='w')
        self.people_entry = tk.Entry(self.input_frame)
        self.people_entry.pack(fill=tk.X, padx=20, pady=2)

        tk.Label(self.input_frame, text="Building Name:").pack(anchor='w')
        self.building_entry = tk.Entry(self.input_frame)
        self.building_entry.pack(fill=tk.X, padx=20, pady=2)

        tk.Label(self.input_frame, text="Floor Number / Name:").pack(anchor='w')
        self.floor_entry = tk.Entry(self.input_frame)
        self.floor_entry.pack(fill=tk.X, padx=20, pady=2)

        tk.Label(self.input_frame, text="Room Number / Name (Optional):").pack(anchor='w')
        self.room_entry = tk.Entry(self.input_frame)
        self.room_entry.pack(fill=tk.X, padx=20, pady=2)

        self.input_frame.pack(pady=10, fill=tk.X)

        submit_btn = tk.Button(self, text="Submit Report", command=self.submit_report)
        submit_btn.pack(pady=20)

        logout_btn = tk.Button(self, text="Logout", command=self.logout)
        logout_btn.pack(side=tk.BOTTOM, pady=10)

        self.toggle_inputs()

    def toggle_inputs(self):
        if self.status_var.get() == "ON":
            for widget in self.input_frame.winfo_children():
                if isinstance(widget, tk.Entry):
                    widget.config(state=tk.NORMAL)
        else:
            for widget in self.input_frame.winfo_children():
                if isinstance(widget, tk.Entry):
                    widget.config(state=tk.DISABLED)
                    widget.delete(0, tk.END)

    def submit_report(self):
        user_id = self.controller.current_user_id
        if not user_id:
            messagebox.showerror("Error", "Not logged in.")
            return

        status = self.status_var.get()
        people = self.people_entry.get() if status == "ON" else "0"
        building = self.building_entry.get() if status == "ON" else ""
        floor = self.floor_entry.get() if status == "ON" else ""
        room = self.room_entry.get() if status == "ON" else ""

        if self.controller.update_citizen_report(user_id, status, people, building, floor, room):
            if status == "ON":
                messagebox.showinfo("Success", "Report submitted. Tracking is ON.")
                self.controller.show_frame("CitizenActiveScreen")
            else:
                messagebox.showinfo("Success", "Report submitted. Tracking is OFF.")

    def logout(self):
        if messagebox.askyesno("Logout", "Are you sure you want to logout?"):
            if self.controller.current_user_id and self.controller.current_user_id in citizen_reports:
                if citizen_reports[self.controller.current_user_id]['status'] == 'ON':
                    self.controller.set_citizen_off(self.controller.current_user_id)

            self.controller.clear_current_user()
            self.controller.show_frame("StartScreen")
            self.status_var.set("OFF")
            self.toggle_inputs()


class CitizenActiveScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.qr_image_label = tk.Label(self)

        tk.Label(self, text="Tracking Active!", font=("Arial", 16)).pack(pady=10)
        tk.Label(self, text="Show this QR code to rescuers when safe.").pack(pady=5)

        self.qr_image_label.pack(pady=20)

        scan_sim_btn = tk.Button(self, text="Simulate Rescuer Scan", command=self.simulate_scan)
        scan_sim_btn.pack(pady=10)

        update_loc_btn = tk.Button(self, text="Update Location/People", command=lambda: controller.show_frame("CitizenWidgetScreen"))
        update_loc_btn.pack(pady=5)

    def generate_and_display_qr(self):
        user_id = self.controller.current_user_id
        if not user_id:
            print("Error: Cannot generate QR, no user logged in.")
            self.qr_image_label.config(image='')
            self.qr_image_label.image = None
            return

        qr_data = f"rescue_user:{user_id}"

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=6,
            border=4,
        )
        qr.add_data(qr_data)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        self.tk_image = ImageTk.PhotoImage(data=img_byte_arr)

        self.qr_image_label.config(image=self.tk_image)
        self.qr_image_label.image = self.tk_image

    def simulate_scan(self):
        user_id = self.controller.current_user_id
        if not user_id: return

        if self.controller.set_citizen_off(user_id):
            messagebox.showinfo("Rescued", "Your status has been set to OFF (Rescued).")
            widget_frame = self.controller.get_frame("CitizenWidgetScreen")
            widget_frame.status_var.set("OFF")
            widget_frame.toggle_inputs()
            self.controller.show_frame("CitizenWidgetScreen")
        else:
            messagebox.showerror("Error", "Could not update status.")


class GovernmentScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.controller.load_citizen_reports() # Load reports when government screen is created

        self.status_label = tk.Label(self, text="Checking fire status...", font=("Arial", 16))
        self.status_label.pack(pady=20)

        self.buildings_frame = tk.Frame(self)
        self.buildings_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        refresh_btn = tk.Button(self, text="Refresh View", command=self.refresh_display)
        refresh_btn.pack(pady=10)

        logout_btn = tk.Button(self, text="Logout", command=self.logout)
        logout_btn.pack(side=tk.BOTTOM, pady=10)

        self.refresh_display()

    def refresh_display(self):
        global fire_announced
        self.controller.load_citizen_reports() # Load fresh data on refresh
        data = self.controller.get_aggregated_fire_data()
        any_on = any(report['status'] == 'ON' for report in citizen_reports.values())
        fire_announced = any_on

        if fire_announced:
            self.status_label.config(text="FIRE ANNOUNCED!", fg="red")
        else:
            self.status_label.config(text="No fire incidents reported.", fg="green")

        for widget in self.buildings_frame.winfo_children():
            widget.destroy()

        if not data:
            tk.Label(self.buildings_frame, text="No active reports.").pack(pady=10)
            return

        tk.Label(self.buildings_frame, text="Buildings with Active Reports:", font=("Arial", 12, "bold")).pack(anchor='w', padx=10)

        for building_name, building_data in data.items():
            total_people = building_data.get('total', 0)
            building_text = f"{building_name}: {total_people} people"
            btn = tk.Button(self.buildings_frame, text=building_text,
                            command=lambda b=building_name: self.show_building_details(b))
            btn.pack(fill=tk.X, padx=20, pady=5)

    def show_building_details(self, building_name):
        detail_frame = self.controller.get_frame("GovernmentBuildingDetailScreen")
        detail_frame.display_details(building_name)
        self.controller.show_frame("GovernmentBuildingDetailScreen")

    def logout(self):
        if messagebox.askyesno("Logout", "Are you sure you want to logout?"):
            self.controller.clear_current_user()
            self.controller.show_frame("StartScreen")

class GovernmentBuildingDetailScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.title_label = tk.Label(self, text="Building Details", font=("Arial", 16))
        self.title_label.pack(pady=10)

        self.details_text = tk.Text(self, wrap=tk.WORD, height=20, width=60)
        self.details_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.details_text.config(state=tk.DISABLED)

        back_btn = tk.Button(self, text="Back to Buildings List", command=lambda: controller.show_frame("GovernmentScreen"))
        back_btn.pack(pady=10)

    def display_details(self, building_name):
        self.controller.load_citizen_reports() # Load fresh data before displaying details
        data = self.controller.get_aggregated_fire_data()
        building_data = data.get(building_name)

        self.title_label.config(text=f"Details for: {building_name}")
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete('1.0', tk.END)

        if not building_data:
            self.details_text.insert(tk.END, "No data available for this building (report might be OFF now).")
        else:
            total_building = building_data.get('total', 0)
            self.details_text.insert(tk.END, f"{building_name}: {total_building} people\n")
            self.details_text.insert(tk.END, "-" * 30 + "\n")

            floors_data = building_data.get('floors', {})
            if not floors_data:
                self.details_text.insert(tk.END, "  No floor details reported.\n")
            else:
                for floor_name, floor_data in sorted(floors_data.items()):
                    total_floor = floor_data.get('total', 0)
                    self.details_text.insert(tk.END, f"  Floor {floor_name}: {total_floor} people\n")

                    rooms_data = floor_data.get('rooms', {})
                    if not rooms_data:
                        self.details_text.insert(tk.END, "    No room details reported for this floor.\n")
                    else:
                        for room_name, room_count in sorted(rooms_data.items()):
                            self.details_text.insert(tk.END, f"  ROOM {room_name}: {room_count} people\n")
                    self.details_text.insert(tk.END, "\n")

        self.details_text.config(state=tk.DISABLED)

# Function to clear the fire data file on exit
def clear_fire_data():
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump({}, f)
        print(f"Fire data in {DATA_FILE} cleared on exit.")
    except Exception as e:
        print(f"Error clearing fire data: {e}")

if __name__ == "__main__":
    app = FireTrackerApp()
    atexit.register(clear_fire_data) # Register the function to be called on exit
    app.mainloop()
