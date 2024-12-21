#include "LSM6DS3.h"
#include <ArduinoBLE.h>
#include "Wire.h"

//Create a instance of class LSM6DS3
LSM6DS3 myIMU(I2C_MODE, 0x6A);    //I2C device address 0x6A

uint16_t SampleRateHz = 100;

BLEService MovementService("19B10010-E8F2-537E-4F6C-D104768A1214");   // create our custom GATT service

// create movement data characteristic (this is a Byte Array) and allow remote device to use Notify
BLECharacteristic moveDataCharacteristic("19B10011-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify, 24, true);

#define DEBUG

void setup() {
  
  #ifdef DEBUG
  Serial.begin(115200);
  while (!Serial);
  #endif

  // begin BLE module initialization
  if (!BLE.begin()) {
    #ifdef DEBUG
    Serial.println("starting BLE failed!");
    #endif
    while (1);
  }


  //Call .begin() to configure the IMUs
  if (myIMU.begin() != 0) {
    #ifdef DEBUG
    Serial.println("Device error");
    #endif
  } else {
    #ifdef DEBUG
    Serial.println("Device OK!");
    #endif
  }

  // set the local name peripheral advertises
  BLE.setLocalName("ktbl");
  // set the UUID for the service this peripheral advertises:
  BLE.setAdvertisedService(MovementService);
  BLE.setAdvertisingInterval(50); // in units of  0.625 ms

  // add the characteristics to the service
  MovementService.addCharacteristic(moveDataCharacteristic);
  
  // add the GATT service
  BLE.addService(MovementService);

  // Create a temporary array for the movement data
  uint8_t mArray[6];
  memset(mArray, '\0', 6);
  moveDataCharacteristic.writeValue(mArray, 6);

  // start advertising
  BLE.advertise();

  #ifdef DEBUG
  Serial.println("Bluetooth device active, waiting for connections...");
  #endif
    // wait for a BLE central
 
}

void loop() {
    unsigned long currentMillis = millis();
    int i=0;
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
      i=i+1;

      updateMovementData();
      currentMillis = millis();
      if (i>100){
        Serial.println("100 updates");
        i=0;
       
      }
        }
    #ifdef DEBUG
    // when the central disconnects, turn off the LED:
    Serial.print("Disconnected from central: ");
    Serial.println(central.address());
    #endif
  }

}

void updateMovementData() {

  //float Mvmnt[3];
  //memset(Mvmnt, '\0', 3);
  uint8_t BLE_mArray[6];//orig 6 uint_8
  memset(BLE_mArray, '\0', 6);

  if (myIMU.begin() == 0) {
    //IMU.readAcceleration(Mvmnt[0], Mvmnt[1], Mvmnt[2]);
    // multiply acceleration by 1000 to convert float in int with decent resolution
    int x_acc=1000*myIMU.readFloatAccelX();
    int y_acc=1000*myIMU.readFloatAccelY();
    int z_acc=1000*myIMU.readFloatAccelZ();
    memcpy(&BLE_mArray[0], &x_acc, 2);
    memcpy(&BLE_mArray[2], &y_acc, 2);
    memcpy(&BLE_mArray[4], &z_acc, 2);
    moveDataCharacteristic.writeValue(BLE_mArray, 6);
    
    /*
    #ifdef DEBUG
    //Serial.print("Ax ");
    Serial.print(x_acc);
    Serial.print(", ");
    Serial.print(y_acc);
    Serial.print(", ");
    Serial.print(z_acc);
    Serial.print("\n");
    #endif
    */
  }

  #ifdef DEBUG
  Serial.flush();
  #endif
  
}
