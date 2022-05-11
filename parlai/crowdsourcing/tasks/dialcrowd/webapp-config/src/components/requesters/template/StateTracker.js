/*********************************************
* @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee 
* Carnegie Mellon University 2022
*********************************************/

import React, {useState, createContext} from "react";

export const StateTracker = createContext();

export const StateTrackerProvider = props => {
    const [generic_introduction, setGenericIntroduction] = useState("");

    return (
        <StateTracker.Provider value={[generic_introduction, setGenericIntroduction]}>
            {props.children}
        </StateTracker.Provider>
    )
}