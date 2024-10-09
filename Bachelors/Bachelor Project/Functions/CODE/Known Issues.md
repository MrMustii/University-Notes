## Bluetooth 
1. Fix the scanning filter such that to filter devices using attributes other than name 
2. In [[ConnectionManager]], when connecting to a device, Remove the busy waiting. This is there to ensure that no other operation can happen while connecting to a device.  The connect operation is not added to the queue because  if there is not connected device the function will not execute the operation. 
3. In general, in the fragments and activity classes , check the initialization of variables and the handling of stopping or destroying the activity/fragment.( Make sure that  life cycle is good)
## Data Processing
1. Never killing the worker thread, to release the resources. . Currently using Handlers to create the thread that the worker will be working on and never killing them.
2. Look for other possible options than background worker, such as foreground service. 
3. Need to re-register the listener in the worker in-case of connecting, disconnecting then reconnecting. 
4. Using doubleArray and Array \<DoubleArray > is probably not efficient to do operations on as other objects. Possibly look into libraries for matrices and vectors.
5. No tests