- Make sure Bluetooth is enabled (Not peams but actual bluetooth)
	- on resume
	- 
- Requesting runtime permissions (Do Each time when you do an action that requires perms )
	- ~~check if you have perms~~
	- ~~Show rationale (Explain why you need the perms )~~
	- ~~Request perms~~ 
	- ~~check user response~~
	- ~~Handel denial of request~~ 
	- And abrupted denial of perms and the sudden turning Bluetooth off (Send notification)
	- 

- Start scanning devices and connecting to them  
	- show scanned devices 
		- **most apps use BLE to connect to a specific type of device because they are designed to perform certain meaningful tasks with only that type of device**
		- the easiest way to make sure an app only ever picks up devices running said custom firmware is to generate a random UUID, and have the firmware advertise this UUID.
		- Warning: a device implementing Bluetooth 4.2’s LE Privacy feature will randomize its public MAC address periodically, so **a MAC address obtained via scanning should not generally be used as a long-term means to identify a device**
	 - TODO
		 - Scan filter 
		 - Make perms part of BLEManager
		 - make sure that if BT is disabled or perms are revoked at any time that the scan does not break
		 - make a view to show scan result 
		 - see and connect to the choosen device 
		 - understand his app
		 - 
	- TORESEARCH

		- 
	- let user pick a device to connect to and bond 




Services 00001801-0000-1000-8000-00805f9b34fb 
 Characteristics 
 | --00002a05-0000-1000-8000-00805f9b34fb
 | --00002b29-0000-1000-8000-00805f9b34fb
 | --00002b2a-0000-1000-8000-00805f9b34fb
 
Services 00001800-0000-1000-8000-00805f9b34fb 
 Characteristics
| --00002a00-0000-1000-8000-00805f9b34fb
| --00002a01-0000-1000-8000-00805f9b34fb
| --00002a04-0000-1000-8000-00805f9b34fb








/////////////////
TODO
- Output the reading  
- fix permas
- fix concurnecy 
- on destroy 
