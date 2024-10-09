#CLass
#### Variables
- [[BLEManager]]
- BluetoothAdapter
- [[ScanAdapter]] 
	- [[ConnectionManager]]
- 

#### Functions 
* Context.[[iisBTAllowed]]
	* Input: 
	* Output: Boolean
	* Description: 
		* Checks if the perms (scan and connect) are granted
	
* Activity.[[requestBTPerms]] 
	* Input: BluetoothAdapter
	* Output: Boolean
	* Description: 
		* first checks if BT is enabled and return false if not enabled 
		* Second runs [[iisBTAllowed]] if not allowed then it requests perms 
		* returns [[iisBTAllowed]] 
	
*  onRequestPermissionsResult (Call back)
	* Input: requestCode (int), permissions (Array \<Strings\>) , grantResult (IntArray)
	* Description:
		* checks the grantresult to see if perms are accepted **SOMEHOW**
		* if contains permanentDennial send to settings 
		* if not permanent show rational again and ask again
		* if all granted and else   ????
## TODO
- ~~In scan adapter check for perms when trying to connect~~
- Make sure that fragment life cycle is good
- ~~onRequestPermissionsResult fix the first when statement~~ 
- onRequestPermissionsResult understand grantresult
- ~~onRequestPermissionsResult figure out all granted and else~~