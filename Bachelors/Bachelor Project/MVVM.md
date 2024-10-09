USING LOGIN AS EXAMPLE

- Features `Auth`
	- Data
		- Local
			- Local database `RoomDB` 
		- Remote
			- apicalls `AuthAPI` as an interface to call api
		- Repository
			- Repository implement the interface `AuthRrepository` as `AuthRrepositoryImpl`   this does all the data handling "Login logic"
	- Presentation
		- Screen `Login` 
			- view
			- view Model 
				- takes Repository `AuthRepository` as input??? and initialize it to use the function 
			- userEvent(Maybe)
	- Domain
		- Use cases 
			- a use case `Verify pass`
		- repository 
			- interface of repository `AuthRrepository` 

shared view model
	-  using a shared view model put all screens in one thier own folder but the viewmodel stays outside
	-  only use shared view model in views of the same feature 

2 features edit the same repo
- have one of the features have access to an object of the interface like a view model
---
- UI
	- ViewModels
	- Views
- Model
	- ss
- Data
	- SQLliteDatabase
	- Remote Database
	- 