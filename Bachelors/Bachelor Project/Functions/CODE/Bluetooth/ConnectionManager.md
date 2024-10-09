#CLass 
#### Variables
- operationQueue  is a Concurrent Linked queue 
- pendingOperation is of type [[BleOperationType]] 
- connecting is Boolean
- deviceGattMap is a hashmap of touple of device and gatt

#### CHECKER Functions
##### These are Boolean function used to check some stuff 
- containsProperty 
	- Checks if the characteristic has a certain property 
- isReadable() 
	- Checks if the characteristic  has the readable property
- isWritable()
	- Checks if the characteristic has the write property 
- isWritableWithoutResponse 
	- Checks if the characteristic has the write without response property
- isIndicatable 
	- Checks if the characteristic has the indicate property
- isNotifiable
	- Checks if the characteristic has the notify property

- findCharacteristic and findDescriptor gets the Characteristic or Descriptor from the uuid  if the device offer it 

#### Enqueue Function 
##### These functions are used to read or write information from the BT device from all parts of the code 

- [[connect]] 
	- Input: device and context 
	- Description:
		- if the devicegattmap does have a device it means that the app is connected to bt device if not it will enqueue the connect operation
-  [[disconnect]] 
	- Input: device  
	- Description:
		- if the devicegattmap does not have a device it means that the app is not connected therefore no enqueuing will happen otherwise a disconnect operation will be enqueued 
-  [[readCharacteristic]] 
	- Input: device and characteristic
	- Description:
		- checks if device is connected and has the readable property if yes enqueue the readcharacreistic function
-  [[readCharacteristic]] 
	- Input: device and characteristic
	- Description:
		- checks if device is connected and has the readable property if yes enqueue the readcharacreistic function

-  [[writeCharacteristic]] 
	- Input: device, characteristic and value
	- Description:
		- checks if device is connected and has the write or write without response property if yes enqueue the write function with the value
-  [[readDescriptor]] 
	- Input: device, characteristic and descriptor
	- Description:
		- checks if device is connected and has the readable property if yes enqueue the ReadDescriptor function
-  [[writeDescriptor]] 
	- Input: device, characteristic, descriptor and value
	- Description:
		- checks if device is connected and has the write permission if yes enqueue the write function with the value
		- 
-  [[enableNotifications]] 
	- Input: device and  characteristic
	- Description:
		- checks if device is connected and has the notifiable or indicatable  if yes enqueue the enable Notifications function
-  [[disableNotifications]] 
	- Input: device and  characteristic
	- Description:
		- checks if device is connected and has the notifiable or indicatable  if yes enqueue the disable Notifications function
- [[requestMtu]]
	- Input: device and mtu value
	- Description:
		- if a device is connected enqueue a  request mtu operation
#### Functions
- [[servicesOnDevice]] 
	- Input: Device 
	- Output: List of gatt services 
	- Description: 
		- gets the services from the gatt of the device 
- [[getBatteryChar]]
	- Input: Device 
	- Output: BluetoothGattCharacteristic
	- Description: 
		- it will try and find the Characteristic for the battery and return it otherwise it will return null
- [[getLightChar]]
	- Input: Device 
	- Output: BluetoothGattCharacteristic
	- Description: 
		- it will try and find the Characteristic for the Light sensor and return it otherwise it will return null
- [[enqueueOperation]] 
	- Input: operation 
	- Output: 
	- Description: 
		- Adds the operation to the queue, if no pending operation calls [[doNextOperation]] 
	
- [[signalEndOfOperation]] 
	- Input:  
	- Output: 
	- Description: 
		- sets the pending operation to null and if the queue is not empty call [[doNextOperation]]
- [[doNextOperation]] 
	- Input:  
	- Output: 
	- Description: 
		- first checks if there is an operation to do 
		- pop the an operation from the queue and sets as the pending operation  
		- if the operation is connect then connect the device and be stuck in a busy waiting situation untill the connection is completed 
		- Save the gatt 
		- here we match one each operation
		- Disconnect
			- close the connection and remove the device from the map
		- CharacteristicWrite
			- find the characteristic and if found call the `gatt.writeCharacteristic` otherwise log and error
		- CharacteristicRead
			- find the characteristic and if found call the `gatt.readCharacteristic` otherwise log and error
		- WriteDescriptor
			- find the Descriptor and if found call the `gatt.writeDescriptor` otherwise log and error
		- ReadDescriptor
			- find the Descriptor and if found call the `gatt.readDescriptor` otherwise log and error
		- MtuRequest
			- call `requestMtu`
		- enableNotifications
			- try to find the characteristic
			- see if characteristic is notifiable if not see if it is indicatable else log error
			- then call `setCharacteristicNotification` where  
		- disableNotifications
			- try to find the characteristic
			- call `setCharacteristicNotification` to disable it   

## TODO
- NO BUSY WAITING
- ~~[[servicesOnDevice]] is not used since we added the [[getBatteryChar]] and [[getLightChar]]  (consider removing it )~~
- ~~all the functions that add stuff to the queue should call [[iisBTAllowed]]~~
- ~~remove endof operation in matching and add them in the callback~~
- we should remove everything we dont use 
- ~~requestMtu maybe needs to be removed since we dont want to risk enqueuing  it out of order instead just increase mtu on connections~~ 
- maybe use the checker operations lol
- 