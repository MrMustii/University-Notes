#Class
# NOTE:
##### This is to represent Bluetooth operations as an objects to add to a queue so that operations don't interfere with each other. All of the operations are subclasses of a sealed class that has an abstract value device  
---
#### Constants
- GATT_MAX_MTU_SIZE


#### Operation
- Connect
	- Takes Device and Context
- Disconnect
	- Takes Device
- DoneConnect
	- Takes Device
- CharacteristicRead 
	- Takes Device and UUID
- CharacteristicWrite
	- Takes Device, UUID, the value to write and write type(with or without response)
- WriteDescriptor
	- Takes Device, UUID of characteristic, UUID of descriptor and the value to write 
- ReadDescriptor
	- Takes Device, UUID of characteristic, UUID of descriptor
- MtuRequest
	- Takes Device and MTU value
- enableNotifications
	- Takes Device and UUID of characteristic
- disableNotifications
	- Takes Device and UUID of characteristic

## TODO
- ~~DoneConnect is not used maybe delete~~
- 