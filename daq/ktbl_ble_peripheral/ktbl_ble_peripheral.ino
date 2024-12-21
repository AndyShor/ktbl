/**
 *  @filename   :   ktbl_ble_peripheral.ino
 *  @brief      :   This software example creates a BLE peripheral with 
 *                  a custom service that contains 2 custom characteristics 
 *                  1. Central to use NOTIFY to receive Acceleration data
 *                  2. Central to use READ to receive Acceleration/Gyro sample rate. They are made to be the same
 *                     Central can also use WRITE to change the sampling rate.
 *
 *  @hardware   :   Arduino Nano 33 Sense BLE  LSM9DS1 IMU sensor
 *  @hardware   :   Arduino Nano 33 Sense rev2 BLE BMI270 IMU sensor
 *  
 *  @author     :   AndyShor
 *
 *  Copyright (C) Andyshor for Application     17 July 2024
  *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documnetation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to  whom the Software is
 * furished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS OR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */


#include <ArduinoBLE.h>
//Arduino nano BLE sense 
//#include <Arduino_LSM9DS1.h>
//Arduino nano BLE sense Rev 2
#include <Arduino_BMI270_BMM150.h>

//#define DEBUG

uint16_t SampleRateHz = 100;

BLEService MovementService("19B10010-E8F2-537E-4F6C-D104768A1214");   // create our custom GATT service


// create movement data characteristic (this is a Byte Array) and allow remote device to use Notify
BLECharacteristic moveDataCharacteristic("19B10011-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify, 24, true);

// create acceleration & gyro sample rate (Hz) characteristic and allow remote device to get notifications
BLEShortCharacteristic SampleRateCharacteristic("19B10012-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite);


void setup() {
  #ifdef DEBUG
  Serial.begin(38400);
  while (!Serial);
  #endif
  
  
  // begin BLE module initialization
  if (!BLE.begin()) {
    #ifdef DEBUG
    Serial.println("starting BLE failed!");
    #endif
    while (1);
  }


  // begin IMU module initialization
  if (!IMU.begin()) {
    #ifdef DEBUG
    Serial.println("Failed to initialize IMU!");
    #endif
    while (1);
  }


  SampleRateHz = (uint16_t)IMU.accelerationSampleRate();
  Serial.println(SampleRateHz);


  if (SampleRateHz != (uint16_t)IMU.gyroscopeSampleRate()) {
    #ifdef DEBUG
    Serial.print(F("Gyroscope sample is not the same as Accelerometer sample rate = "));
    Serial.print(SampleRateHz);
    Serial.println(" Hz");
    #endif
  }
  else {
    #ifdef DEBUG
    Serial.print(F("The Acceleration & Gyro sample rate is "));
    Serial.print(SampleRateHz);
    Serial.println(" Hz");
    #endif
  }
  #ifdef DEBUG
  Serial.println();
  #endif
  
  // set the local name peripheral advertises
  BLE.setLocalName("ktbl");
  // set the UUID for the service this peripheral advertises:
  BLE.setAdvertisedService(MovementService);
  BLE.setAdvertisingInterval(32); // in units of  0.625 ms


  // add the characteristics to the service
  MovementService.addCharacteristic(moveDataCharacteristic);
  MovementService.addCharacteristic(SampleRateCharacteristic);


  // add the GATT service
  BLE.addService(MovementService);


  // Create a temporary array for the movement data
  uint8_t mArray[6];
  memset(mArray, '\0', 6);
  moveDataCharacteristic.writeValue(mArray, 6);


  SampleRateCharacteristic.writeValue(SampleRateHz);

  // set the interval for the communication
  //BLE.setConnectionInterval(12, 12); // 1/(4 * 1.25ms) 200 Hz
  // start advertising
  BLE.advertise();


  #ifdef DEBUG
  Serial.println("Bluetooth device active, waiting for connections...");
  #endif
    // wait for a BLE central
  
}


void loop() {
  BLEDevice central = BLE.central();

  // if a central is connected to the peripheral:
  if (central) {
    #ifdef DEBUG
    Serial.print("Connected to central: ");
    // print the central's BT address:
    Serial.println(central.address());
    #endif

    // check the battery level every 200ms
    // while the central is connected:
    while (central.connected()) {
      // Update the BLE data
      updateMovementData();
    }
    #ifdef DEBUG
    // when the central disconnects, turn off the LED:
    Serial.print("Disconnected from central: ");
    Serial.println(central.address());
    #endif
  }
}


void updateMovementData() {
  /* Read the current data from the Arduino LSM9DS1 sensor.
  */
  float Mvmnt[3];
  memset(Mvmnt, '\0', 3);
  uint8_t BLE_mArray[6];//orig 6 uint_8
  memset(BLE_mArray, '\0', 6);

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(Mvmnt[0], Mvmnt[1], Mvmnt[2]);
    // multiply acceleration by 1000 to convert float in int with decent resolution
    int x_acc=1000*Mvmnt[0];
    int y_acc=1000*Mvmnt[1];
    int z_acc=1000*Mvmnt[2];
    memcpy(&BLE_mArray[0], &x_acc, 2);
    memcpy(&BLE_mArray[2], &y_acc, 2);
    memcpy(&BLE_mArray[4], &z_acc, 2);
    moveDataCharacteristic.writeValue(BLE_mArray, 6);
    #ifdef DEBUG
    //Serial.print("Ax ");
    Serial.print(x_acc);
    Serial.print(", ");
    Serial.print(y_acc);
    Serial.print(", ");
    Serial.print(z_acc);
    Serial.print("\n");
    #endif
  }

  #ifdef DEBUG
  Serial.flush();
  #endif



  
}
