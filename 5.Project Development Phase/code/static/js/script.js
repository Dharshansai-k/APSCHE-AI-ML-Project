const form = document.getElementById("predictionForm");

form.addEventListener("submit", async function(e){

    e.preventDefault();

    const result = document.getElementById("result");

    result.style.display = "block";
    result.className = "";

    result.innerHTML = `
        <h4>🔄 Predicting...</h4>
    `;

    const data = {

        N: document.getElementById("N").value,

        P: document.getElementById("P").value,

        K: document.getElementById("K").value,

        temperature: document.getElementById("temperature").value,

        humidity: document.getElementById("humidity").value,

        ph: document.getElementById("ph").value,

        rainfall: document.getElementById("rainfall").value

    };

    try{

        const response = await fetch("/predict",{

            method:"POST",

            headers:{
                "Content-Type":"application/json"
            },

            body:JSON.stringify(data)

        });

        const res = await response.json();

        if(res.success){
                result.className = "success";

                result.innerHTML = `

                <h2>${res.info.emoji} ${res.crop.toUpperCase()}</h2>

                <hr>

                <h4>🎯 Confidence : ${res.confidence}%</h4>

                <p><strong>💧 Water Requirement:</strong> ${res.info.water}</p>

                <p><strong>🌡️ Temperature:</strong> ${res.info.temperature}</p>

                <p>${res.info.description}</p>

                `;
        }

        else{

            result.className="error";

            result.innerHTML=`

                <h4>❌ ${res.message}</h4>

            `;

        }

    }

    catch(err){

        result.className="error";

        result.innerHTML=`

            <h4>

            Server Error

            </h4>

        `;

    }

});

document.getElementById("resetBtn").addEventListener("click", () => {

    document.getElementById("predictionForm").reset();

    const result = document.getElementById("result");

    result.style.display = "none";

});