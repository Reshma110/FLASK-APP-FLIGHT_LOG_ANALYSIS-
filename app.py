# from flask import Flask, render_template, request
# import pandas as pd
# from pymavlink import mavutil
# import os
# import numpy as np
# import datetime
# from werkzeug.utils import secure_filename

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'bin'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def safe_times(df, label="UNKNOWN"):
#     for col in ['TimeUS', 'time_boot_ms', 'TimeMS']:
#         if col in df.columns and df[col].notnull().any():
#             try:
#                 if 'TimeUS' in col:
#                     return pd.to_numeric(df[col], errors='coerce') / 1e6
#                 else:
#                     return pd.to_numeric(df[col], errors='coerce') / 1e3
#             except Exception as e:
#                 print(f"[ERROR] TimeS for {label} failed: {e}")
#                 raise
#     raise ValueError(f"No usable timestamp in {label}")

# def load_df_generic(bin_path, msg_type, label, inst_col=None):
#     mlog = mavutil.mavlink_connection(bin_path, dialect='ardupilotmega')
#     msgs = []
#     while True:
#         msg = mlog.recv_match(type=msg_type, blocking=False)
#         if msg is None:
#             break
#         msgs.append(msg.to_dict())
#     if not msgs:
#         raise RuntimeError(f"No {msg_type} messages found")
#     df = pd.DataFrame(msgs)
#     df['TimeS'] = safe_times(df, label=label)
#     if inst_col:
#         df[inst_col] = pd.to_numeric(df.get(inst_col, 0), errors='coerce').fillna(0).astype(int)
#     return df

# # Loaders
# def load_baro_df(path): return load_df_generic(path, 'BARO', 'BARO', 'I')
# def load_rcou_df(path): return load_df_generic(path, 'RCOU', 'RCOU')
# def load_powr_df(path): return load_df_generic(path, 'POWR', 'POWR')
# def load_att_df(path): return load_df_generic(path, 'ATT', 'ATT')
# def load_rate_df(path): return load_df_generic(path, 'RATE', 'RATE')
# def load_vibe_df(path): return load_df_generic(path, 'VIBE', 'VIBE', 'IMU')
# def load_pscd_df(path): return load_df_generic(path, 'PSCD', 'PSCD')
# def load_psce_df(path): return load_df_generic(path, 'PSCE', 'PSCE')
# def load_pscn_df(path): return load_df_generic(path, 'PSCN', 'PSCN')
# def load_imu_df(path): return load_df_generic(path, 'IMU', 'IMU')
# def load_mag_df(path): return load_df_generic(path, 'MAG', 'MAG', 'I')
# def load_rcin_df(path): return load_df_generic(path, 'RCIN', 'RCIN')
# def load_gpa_df(path): return load_df_generic(path, 'GPA', 'GPA', 'I')
# def load_bat_df(path): return load_df_generic(path, 'BAT', 'BAT', 'Inst')

# # Summary
# def compute_summary_data(path, imu_df, att_df, pscd_df,psce_df,pscn_df):
#     try:
#         start_time = datetime.datetime.fromtimestamp(imu_df["TimeS"].min())
#         end_time = datetime.datetime.fromtimestamp(imu_df["TimeS"].max())
#         duration = end_time - start_time
#         duration_str = str(duration).split('.')[0]

#         max_tilt = max(att_df['Roll'].abs().max(), att_df['Pitch'].abs().max())

#         if not pscd_df.empty and 'VN' in pscn_df.columns and 'VE' in psce_df.columns and 'VD' in pscd_df.columns:
#             speed_squared = pscn_df['VN']**2 + psce_df['VE']**2 + pscd_df['VD']**2
#             speed = np.sqrt(speed_squared)
#             avg_speed = speed.mean() * 3.6
#             max_speed = speed.max() * 3.6
#             max_speed_up = pscd_df['VD'].min() * -1
#             max_speed_down = pscd_df['VD'].max()
#         else:
#             avg_speed = max_speed = max_speed_up = max_speed_down = None

                
#         def compute_distance(df, columns):
#             if all(col in df.columns for col in columns):
#                 return np.sqrt((df[columns].diff()**2).sum(axis=1)).sum()
#             return None
        
#         distance_pscd = compute_distance(pscd_df, ['PD'])
#         distance_psce = compute_distance(psce_df, ['PE'])
#         distance_pscn = compute_distance(pscn_df, ['PN'])
        
#         total_distance = (distance_pscd if distance_pscd is not None else 0) + \
#                          (distance_psce if distance_psce is not None else 0) + \
#                          (distance_pscn if distance_pscn is not None else 0)

#         max_altitude_diff = pscd_df['PD'].max() - pscd_df['PD'].min() if 'PD' in pscd_df.columns else None

#         summary = {
#             "Airframe": "hexa",
#             "Hardware": "ARDUPILOT",
#             "Software Version": "v1.13.2 (46a12a09)",
#             "OS Version": "NuttX, v11.0.0",
#             "Estimator": "EKF2",
#             "Logging Start": start_time,
#             "Logging Duration": duration_str,
#             "Flight Time": duration_str,
#             "Vehicle UUID": "000600000000393137313132510100280040",
#             "Total Distance": f"{total_distance:.1f} m" if total_distance else "N/A",
#             "Max Altitude Difference": f"{max_altitude_diff:.1f} m" if max_altitude_diff else "N/A",
#             "Average Speed": f"{avg_speed:.1f} km/h" if avg_speed else "N/A",
#             "Max Speed": f"{max_speed:.1f} km/h" if max_speed else "N/A",
#             "Max Tilt Angle": f"{max_tilt:.1f} deg" if max_tilt else "N/A"
#         }

#         return summary
#     except Exception as e:
#         print(f"[ERROR] Summary computation failed: {e}")
#         return {}

# # Plot Helpers
# def create_channel_plot(df, prefix, label_prefix):
#     plot_data = []
#     for ch in [f'{prefix}{i}' for i in range(1, 15) if f'{prefix}{i}' in df.columns]:
#         plot_data.append({
#             'x': df['TimeS'].tolist(),
#             'y': pd.to_numeric(df[ch], errors='coerce').tolist(),
#             'mode': 'lines',
#             'name': f'{label_prefix}: {ch}'
#         })
#     return plot_data

# def create_instance_plot(df, params, id_col='I', label_prefix=''):
#     plot_data = []
#     for param in params:
#         for inst in sorted(df[id_col].unique()):
#             subset = df[df[id_col] == inst]
#             if param in subset.columns:
#                 plot_data.append({
#                     'x': subset['TimeS'].tolist(),
#                     'y': pd.to_numeric(subset[param], errors='coerce').tolist(),
#                     'mode': 'lines',
#                     'name': f'{label_prefix}{param} (Inst {inst})'
#                 })
#     return plot_data

# def create_basic_plot(df, params, label_prefix=''):
#     plot_data = []
#     for param in params:
#         if param in df.columns:
#             plot_data.append({
#                 'x': df['TimeS'].tolist(),
#                 'y': pd.to_numeric(df[param], errors='coerce').tolist(),
#                 'mode': 'lines',
#                 'name': f'{label_prefix}{param}'
#             })
#     return plot_data

# @app.route('/')
# def home():
#     return render_template('upload.html')

# @app.route('/upload', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         file = request.files.get('filearg')
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(path)

#             try:
#                 baro_df = load_baro_df(path)
#                 baro_df_instance_0 = baro_df[baro_df['I'] == 0]
#                 baro_df_instance_1 = baro_df[baro_df['I'] == 1]
#                 rcou_df = load_rcou_df(path)
#                 powr_df = load_powr_df(path)
#                 att_df = load_att_df(path)
#                 rate_df = load_rate_df(path)
#                 vibe_df = load_vibe_df(path)
#                 pscd_df = load_pscd_df(path)
#                 psce_df = load_psce_df(path)
#                 pscn_df = load_pscn_df(path)
#                 imu_df = load_imu_df(path)

#                 summary_data = compute_summary_data(path, imu_df, att_df, pscd_df,psce_df, pscn_df)

#                 try: mag_df = load_mag_df(path); mag_plot_data = create_instance_plot(mag_df, ['MagX','MagY','MagZ','OfsX','OfsY','OfsZ','MOX','MOY','MOZ','Health','S'], 'I', 'MAG: ')
#                 except: mag_plot_data = []
#                 try: rcin_df = load_rcin_df(path); rcin_plot_data = create_channel_plot(rcin_df, 'C', 'RCIN')
#                 except: rcin_plot_data = []
#                 try: gpa_df = load_gpa_df(path); gpa_plot_data = create_instance_plot(gpa_df, ['VDop', 'HAcc', 'VAcc', 'SAcc', 'YAcc', 'VV', 'SMS', 'Delta', 'Und', 'RTCMFU', 'RTCMFD', 'TimeS'], 'I', 'GPA: ')
#                 except: gpa_plot_data = []
#                 try: bat_df = load_bat_df(path); bat_plot_data = create_instance_plot(bat_df, ['Volt','VoltR','Curr','CurrTot','EnrgTot','Temp','Res','RemPct','H','SH'], 'Inst', 'BAT: ')
#                 except: bat_plot_data = []

#                 gyro_data, acc_data = [], []
#                 for param in ['GyrX','GyrY','GyrZ']:
#                     if param in imu_df.columns:
#                         gyro_data.append({'x': imu_df['TimeS'].tolist(), 'y': imu_df[param].tolist(), 'mode': 'lines', 'name': param})
#                 for param in ['AccX','AccY','AccZ','EG','EA','T','GH','AH','GHz','AHz']:
#                     if param in imu_df.columns:
#                         acc_data.append({'x': imu_df['TimeS'].tolist(), 'y': imu_df[param].tolist(), 'mode': 'lines', 'name': param})

#                 baro_params = [c for c in baro_df.columns if c not in ('TimeS', 'I') and pd.api.types.is_numeric_dtype(baro_df[c])]

#                 return render_template('index.html',
#                     summary_data=summary_data,
#                     baro_plot_data_instance_0=create_instance_plot(baro_df_instance_0, baro_params, 'I', 'BARO: '),
#                     baro_plot_data_instance_1=create_instance_plot(baro_df_instance_1, baro_params, 'I', 'BARO: '),
#                     rcou_plot_data=create_channel_plot(rcou_df, 'C', 'RCOU'),
#                     powr_plot_data=create_basic_plot(powr_df, ['Vcc','VServo','Flags','AccFlags','Safety'], 'POWR: '),
#                     att_plot_data=create_basic_plot(att_df, ['Roll','Pitch','Yaw'], 'ATT: '),
#                     rate_plot_data=create_basic_plot(rate_df, ['R', 'P', 'Y', 'RDes', 'PDes', 'YDes', 'ROut', 'POut', 'YOut'], 'RATE: '),
#                     vibe_plot_data=create_instance_plot(vibe_df, ['VibeX','VibeY','VibeZ','Clip'], 'IMU', 'VIBE: '),
#                     pscd_plot_data=create_basic_plot(pscd_df, ['TPD','PD','DVD','TVD','VD','DAD','TAD','AD'], 'PSCD: '),
#                     psce_plot_data=create_basic_plot(psce_df, ['TPE','PE','DVE','TVE','VE','DAE','TAE','AE'], 'PSCE: '),
#                     pscn_plot_data=create_basic_plot(pscn_df, ['TPN','PN','DVN','TVN','VN','DAN','TAN','AN'], 'PSCN: '),
#                     gyro_data=gyro_data,
#                     acc_data=acc_data,
#                     mag_plot_data=mag_plot_data,
#                     rcin_plot_data=rcin_plot_data,
#                     gpa_plot_data=gpa_plot_data,
#                     bat_plot_data=bat_plot_data
#                 )
#             except Exception as e:
#                 return f"<h3>Error processing file: {e}</h3>"
#         return "<h3>Invalid file. Only .bin files allowed.</h3>"
#     return render_template('upload.html')

# if __name__ == '__main__':
#     app.run(debug=True)






from flask import Flask, render_template, request
import pandas as pd
from pymavlink import mavutil
import os
import numpy as np
import datetime
from werkzeug.utils import secure_filename
from collections import OrderedDict
import re

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'bin'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODE_MAP = {
    0: 'Stabilize', 1: 'Acro', 2: 'AltHold', 3: 'Auto', 4: 'Guided', 5: 'Loiter', 6: 'Circle',
    7: 'Drift', 8: 'Sport', 9: 'Flip',  10: 'AutoTune',11: 'PosHold', 12: 'Brake',13: 'Throw', 14: 'Avoid_ADSB',
    15: 'Guided_NoGPS', 16: 'SmartRTL', 17: 'FlowHold',
    18: 'Follow', 19: 'ZigZag', 20: 'SystemID',
    21: 'Heli_Autorotate', 22: 'Auto RTL', 23: 'Turtle'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_times(df, label="UNKNOWN"):
    for col in ['TimeUS', 'time_boot_ms', 'TimeMS']:
        if col in df.columns and df[col].notnull().any():
            try:
                return pd.to_numeric(df[col], errors='coerce') / (1e6 if 'TimeUS' in col else 1e3)
            except Exception as e:
                print(f"[ERROR] TimeS for {label} failed: {e}")
                raise
    raise ValueError(f"No usable timestamp in {label}")

def load_df_generic(bin_path, msg_type, label, inst_col=None):
    mlog = mavutil.mavlink_connection(bin_path, dialect='ardupilotmega')
    msgs = []
    while True:
        msg = mlog.recv_match(type=msg_type, blocking=False)
        if msg is None:
            break
        msgs.append(msg.to_dict())
    if not msgs:
        raise RuntimeError(f"No {msg_type} messages found")
    df = pd.DataFrame(msgs)
    df['TimeS'] = safe_times(df, label=label)
    if inst_col:
        df[inst_col] = pd.to_numeric(df.get(inst_col, 0), errors='coerce').fillna(0).astype(int)
    return df

# Loaders
def load_mode_df(path):
    df = load_df_generic(path, 'MODE', 'MODE')
    df['ModeName'] = df['Mode'].map(MODE_MAP).fillna(df['Mode'].astype(str))
    return df

def load_baro_df(path): return load_df_generic(path, 'BARO', 'BARO', 'I')
def load_rcou_df(path): return load_df_generic(path, 'RCOU', 'RCOU')
def load_powr_df(path): return load_df_generic(path, 'POWR', 'POWR')
def load_att_df(path): return load_df_generic(path, 'ATT', 'ATT')
def load_rate_df(path): return load_df_generic(path, 'RATE', 'RATE')
def load_vibe_df(path): return load_df_generic(path, 'VIBE', 'VIBE', 'IMU')
def load_pscd_df(path): return load_df_generic(path, 'PSCD', 'PSCD')
def load_psce_df(path): return load_df_generic(path, 'PSCE', 'PSCE')
def load_pscn_df(path): return load_df_generic(path, 'PSCN', 'PSCN')
def load_imu_df(path): return load_df_generic(path, 'IMU', 'IMU')
def load_mag_df(path): return load_df_generic(path, 'MAG', 'MAG', 'I')
def load_rcin_df(path): return load_df_generic(path, 'RCIN', 'RCIN')
def load_gpa_df(path): return load_df_generic(path, 'GPA', 'GPA', 'I')
def load_bat_df(path): return load_df_generic(path, 'BAT', 'BAT', 'Inst')

def load_msg_df(path):
    mlog = mavutil.mavlink_connection(path, dialect='ardupilotmega')
    msgs = []
    while True:
        msg = mlog.recv_match(type='MSG', blocking=False)
        if msg is None:
            break
        msgs.append(msg.to_dict())
    return pd.DataFrame(msgs)


def create_channel_plot(df, prefix, label_prefix):
    return [{
        'x': df['TimeS'].tolist(),
        'y': pd.to_numeric(df[ch], errors='coerce').tolist(),
        'mode': 'lines',
        'name': f'{label_prefix}: {ch}'
    } for ch in [f'{prefix}{i}' for i in range(1, 15) if f'{prefix}{i}' in df.columns]]

def create_instance_plot(df, params, id_col='I', label_prefix=''):
    plots = []
    for param in params:
        for inst in sorted(df[id_col].unique()):
            subset = df[df[id_col] == inst]
            if param in subset.columns:
                plots.append({
                    'x': subset['TimeS'].tolist(),
                    'y': pd.to_numeric(subset[param], errors='coerce').tolist(),
                    'mode': 'lines',
                    'name': f'{label_prefix}{param} (Inst {inst})'
                })
    return plots

def create_basic_plot(df, params, label_prefix=''):
    return [{
        'x': df['TimeS'].tolist(),
        'y': pd.to_numeric(df[param], errors='coerce').tolist(),
        'mode': 'lines',
        'name': f'{label_prefix}{param}'
    } for param in params if param in df.columns]

def compute_summary_data(path, imu_df, att_df, pscd_df, psce_df, pscn_df,log_start_datetime=None):
    try:
        start_time = log_start_datetime or datetime.datetime.fromtimestamp(imu_df["TimeS"].min())
        start_time_1 = datetime.datetime.fromtimestamp(imu_df["TimeS"].min())

        end_time = datetime.datetime.fromtimestamp(imu_df["TimeS"].max())
        duration = datetime.timedelta(seconds=imu_df["TimeS"].max() - imu_df["TimeS"].min())
        end_time_1 = start_time + duration


        duration_str = str(end_time - start_time_1).split('.')[0]
        max_tilt = max(att_df['Roll'].abs().max(), att_df['Pitch'].abs().max())
        if not pscd_df.empty and 'VN' in pscn_df and 'VE' in psce_df and 'VD' in pscd_df:
            speed = np.sqrt(pscn_df['VN']**2 + psce_df['VE']**2 + pscd_df['VD']**2)
            avg_speed = speed.mean() * 3.6
            max_speed = speed.max() * 3.6
        else:
            avg_speed = max_speed = None

        def compute_distance(df, columns):
            if all(col in df.columns for col in columns):
                return np.sqrt((df[columns].diff()**2).sum(axis=1)).sum()
            return None
        distance_pscd = compute_distance(pscd_df, ['PD'])
        distance_psce = compute_distance(psce_df, ['PE'])
        distance_pscn = compute_distance(pscn_df, ['PN'])
        
        total_distance = (distance_pscd if distance_pscd is not None else 0) + \
                        (distance_psce if distance_psce is not None else 0) + \
                (distance_pscn if distance_pscn is not None else 0)

        max_altitude_diff = pscd_df['PD'].max() - pscd_df['PD'].min() if 'PD' in pscd_df else None
        return {
            "Logging Start": start_time,
            "Logging End": end_time_1,
            "Flight Time": duration_str,
            "Max Tilt Angle": f"{max_tilt:.1f} deg",
            "Average Speed": f"{avg_speed:.1f} km/h" if avg_speed else "N/A",
            "Max Speed": f"{max_speed:.1f} km/h" if max_speed else "N/A",
            "Total Distance": f"{total_distance:.1f} m",
            "Max Altitude Difference": f"{max_altitude_diff:.1f} m" if max_altitude_diff else "N/A"
        }
    except Exception as e:
        print(f"[ERROR in summary]: {e}")
        return {}

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('filearg')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            try:
                match = re.search(r'(\d{4}-\d{2}-\d{2})[ _](\d{2}-\d{2}-\d{2})', filename)
                if match:
                    date_part = match.group(1)
                    time_part = match.group(2).replace('-', ':')
                    log_start_datetime = datetime.datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S")

                else:
                    log_start_datetime = "Unknown"
            except:
                log_start_datetime = "Unknown"

            try:
                # Load data
                baro_df = load_baro_df(path)
                rcou_df = load_rcou_df(path)
                powr_df = load_powr_df(path)
                att_df = load_att_df(path)
                rate_df = load_rate_df(path)
                vibe_df = load_vibe_df(path)
                pscd_df = load_pscd_df(path)
                psce_df = load_psce_df(path)
                pscn_df = load_pscn_df(path)
                imu_df = load_imu_df(path)
                mode_df = load_mode_df(path)
                msg_df = load_msg_df(path) 
                firmware_msgs = msg_df['Message'].iloc[[0, 1, 2, 9]].dropna().tolist()

                mode_df = mode_df[['TimeS', 'ModeName']].dropna().sort_values('TimeS')
                mode_segments = []
                for i in range(len(mode_df) - 1):
                    mode_segments.append({
                        'start': mode_df.iloc[i]['TimeS'],
                        'end': mode_df.iloc[i + 1]['TimeS'],
                        'mode': mode_df.iloc[i]['ModeName']
                    })

                summary_data = compute_summary_data(path, imu_df, att_df, pscd_df, psce_df, pscn_df, log_start_datetime)
                summary_data=OrderedDict([("Vehicle & Firmware Info", "\n".join(firmware_msgs)),*summary_data.items()])
     


                baro_df_0 = baro_df[baro_df['I'] == 0]
                baro_df_1 = baro_df[baro_df['I'] == 1]
                baro_params = [c for c in baro_df.columns if c not in ('TimeS', 'I') and pd.api.types.is_numeric_dtype(baro_df[c])]

                gyro_data = create_basic_plot(imu_df, ['GyrX','GyrY','GyrZ'], '')
                acc_data = create_basic_plot(imu_df, ['AccX','AccY','AccZ'], '')

                try: mag_df = load_mag_df(path); mag_plot_data = create_instance_plot(mag_df, ['MagX','MagY','MagZ','OfsX','OfsY','OfsZ','MOX','MOY','MOZ','Health','S'], 'I', 'MAG: ')
                except: mag_plot_data = []
                try: rcin_df = load_rcin_df(path); rcin_plot_data = create_channel_plot(rcin_df, 'C', 'RCIN')
                except: rcin_plot_data = []
                try: gpa_df = load_gpa_df(path); gpa_plot_data = create_instance_plot(gpa_df, ['VDop', 'HAcc', 'VAcc', 'SAcc', 'YAcc', 'VV', 'SMS', 'Delta', 'Und', 'RTCMFU', 'RTCMFD'], 'I', 'GPA: ')
                except: gpa_plot_data = []
                try: bat_df = load_bat_df(path); bat_plot_data = create_instance_plot(bat_df, ['Volt','VoltR','Curr','CurrTot','EnrgTot','Temp','Res','RemPct','H','SH'], 'Inst', 'BAT: ')
                except: bat_plot_data = []
                summary_data["Logging Start"] = summary_data["Logging Start"].strftime("%Y-%m-%d %H:%M:%S")
                summary_data["Logging End"] = summary_data["Logging End"].strftime("%Y-%m-%d %H:%M:%S")

                return render_template('index.html',
                    summary_data=summary_data,
                    baro_plot_data_instance_0=create_instance_plot(baro_df_0, baro_params, 'I', 'BARO: '),
                    baro_plot_data_instance_1=create_instance_plot(baro_df_1, baro_params, 'I', 'BARO: '),
                    rcou_plot_data=create_channel_plot(rcou_df, 'C', 'RCOU'),
                    powr_plot_data=create_basic_plot(powr_df, ['Vcc','VServo'], 'POWR: '),
                    att_plot_data=create_basic_plot(att_df, ['Roll','Pitch','Yaw'], 'ATT: '),
                    rate_plot_data=create_basic_plot(rate_df, ['R', 'P', 'Y'], 'RATE: '),
                    vibe_plot_data=create_instance_plot(vibe_df, ['VibeX','VibeY','VibeZ'], 'IMU', 'VIBE: '),
                    pscd_plot_data=create_basic_plot(pscd_df, ['PD','VD'], 'PSCD: '),
                    psce_plot_data=create_basic_plot(psce_df, ['PE','VE'], 'PSCE: '),
                    pscn_plot_data=create_basic_plot(pscn_df, ['PN','VN'], 'PSCN: '),
                    gyro_data=gyro_data,
                    acc_data=acc_data,
                    mag_plot_data=mag_plot_data,
                    rcin_plot_data=rcin_plot_data,
                    gpa_plot_data=gpa_plot_data,
                    bat_plot_data=bat_plot_data,
                    mode_segments=mode_segments
                )

            except Exception as e:
                return f"<h3>Error: {e}</h3>"

        return "<h3>Only .bin files allowed</h3>"

    return render_template('upload.html')

@app.route('/')
def home():
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
