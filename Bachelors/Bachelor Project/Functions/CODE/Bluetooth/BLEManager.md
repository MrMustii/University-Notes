#CLass
#### Constants
- CCC_DESCRIPTOR_UUID
- LightServiceUUID
- LightCharacteristicUUID
- BatteryServiceUUID
- BatteryCharacteristicUUID


#### Variables
- SCAN_PERIOD : Length of scan
- bleScanner : a scanner object 
- scanning: boolean true when scanning  
- handler: runs in ul thread and 
- filter : how to filter the scan
- scanSettings: the settings for the scan
#### Functions
- [[scanBLE]] 
	- Input
	- Output
	- Description: 
		- Checks for perms
		- Clear results and update UI 
		- If not scanning start scanning using the filter and settings and scancallback
		- After the time period stop scanning
		- if no perms or try to scan while it is already scanning put a TOAST message
- [[scanCallback]]
	- `onScanResult`
		- Input: callbackType and result
		- Output
		- Description:
			- each time a device is scanned if it is already saved in the scanResults list then update the list (in case the name has changed since you only compare MAC address) and UI
			- if not already scanned add it to to the list and update UI
## TODO
- Fix the filter to not filter just the name
- [[scanBLE]] Maybe use [[iisBTAllowed]] 
