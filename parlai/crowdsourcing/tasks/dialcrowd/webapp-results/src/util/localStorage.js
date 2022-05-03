/*********************************************
* @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee 
* Carnegie Mellon University 2022
*********************************************/

export const loadState = () => {
    // We need the try block because user may not permit our accessing localStorage.
    try {
      const serializedState = localStorage.getItem('state');
      if (serializedState === null) { // The key 'state' does not exist.
        return undefined;             // Let our reducer initialize the app.
      }
  
      return JSON.parse(serializedState)
  
    } catch (error) {
      console.log(error);
      return undefined // Let our reducer initialize the app.
    }
  };
  
  export const saveState = (state) => {
    try {
      // Serialize the state. Redux store is recommended to be serializable
      const serializedState = JSON.stringify(state);
      localStorage.setItem('state', serializedState)
  
    } catch (error) {
      console.log(error)
    }
  };