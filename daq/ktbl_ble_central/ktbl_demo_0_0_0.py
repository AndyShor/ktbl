import asyncio
from bleak import BleakClient
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
import os.path

# adress of Arduino Nano Sense BLE device
#address = "a6:14:e8:c3:50:fb"

# adress of Arduino Nano Sense BLE rev 2 device
address = "81:46:1d:ab:a4:44"

# adress of seed xiao device 
#address = "85:8A:7A:43:AD:C7"


data_UUID='19b10011-e8f2-537e-4f6c-d104768a1214'
characteristic_UUID=data_UUID

gz_arr=[]
gx_arr=[]
gy_arr=[]
timestamps_arr=[]

timespan=30
test_length=100
#data_folder='raw_data'
#data_folder=os.path.abspath(os.path.join( "..", ".."data"))
data_folder=os.path.join( "..", "..", "data", "raw")
print(data_folder)

async def main(address):
    global i
    client = BleakClient(address)
    now = time.time()

    async def callback(sender, data):
        global i, gz_arr, gx_arr, gy_arr, timestamps_arr
        timestamp = time.time() - now

        timestamps_arr.append(timestamp)
        #times_np[i] = timestamp
        gz_read=int.from_bytes(data[4:6], 'little', signed=True) / 1000
        gz_arr.append(gz_read)
        gy_read = int.from_bytes(data[2:4], 'little', signed=True) / 1000
        gy_arr.append(gy_read)
        gx_read = int.from_bytes(data[0:2], 'little', signed=True) / 1000
        gx_arr.append(gx_read)

        i = i + 1
        if i%10==0:
            print(f"{timestamp}: {gx_read} {gy_read} {gz_read}")


    try:
        await client.connect()
        if client.is_connected:
            print("Connected!, MTU size:{}".format(client.mtu_size))
            await client.start_notify(data_UUID, callback)
            await asyncio.sleep(timespan)
            print("Subscribed to BLE notifications.")
        else:
            print("Having trouble connecting to arduino at ")


    except Exception as err:
        print("Error: {0}.".format(err, type(err)))

    finally:
        await client.disconnect()
        print(' client disconnected')

i = 0

asyncio.run(main(address))
test_length=timespan*120
times_np = np.array(timestamps_arr)
gz_np = np.array(gz_arr)
gy_np = np.array(gy_arr)
gx_np = np.array(gx_arr)
tgF_np = np.sqrt(gz_np**2+gy_np**2+gx_np**2)

filename = os.path.join(data_folder, time.strftime("%Y%m%d-%H%M%S") + '.csv')
f = open(filename, 'w')
writer = csv.writer(f)
data = [[timevalue, g_x, g_y, g_z, tgF] for timevalue, g_x, g_y, g_z, tgF in
            zip(times_np, gy_np, gx_np, gz_np, tgF_np)]
writer.writerows(data)
f.close()

figure, ax = plt.subplots(figsize=(10, 8))
#line3 = ax.plot(times_np,  gx_np, '--bo', color='k')
#line4 = ax.plot(times_np,  gz_np, '--bo', color='b')
line1 = ax.plot(np.arange(len(times_np)), times_np, '--bo')
plt.title("Datastream", fontsize=20)

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

figure, ax = plt.subplots(figsize=(10, 8))
line3 = ax.plot(times_np,  gx_np, '--', color='k')
line4 = ax.plot(times_np,  gz_np, '--', color='b')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()