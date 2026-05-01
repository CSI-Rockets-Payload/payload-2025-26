# import serial.tools.list_ports
# ports = serial.tools.list_ports.comports()
# for port in ports:
#     print(port.device)

import serial
ser = serial.Serial('COM3', 4800, timeout=1)
while True:
    line  = ser.readline().decode('ascii', errors='replace').strip()
    if line.startswith('$GPGGA'):
        print(line)

