// ===============================
// CIWIFITS AI
// Navbar Scroll Effect
// ===============================

const header = document.querySelector(".header");

let lastScroll = 0;

window.addEventListener("scroll", () => {

    const currentScroll = window.pageYOffset;

    // Tambah background saat scroll
    if(currentScroll > 30){

        header.classList.add("scrolled");

    }else{

        header.classList.remove("scrolled");

    }

    // Hide navbar saat scroll ke bawah
    if(currentScroll > lastScroll && currentScroll > 120){

        header.classList.add("hide");

    }else{

        header.classList.remove("hide");

    }

    lastScroll = currentScroll;

});

// ===============================
// Reveal Animation
// ===============================

const reveals = document.querySelectorAll("section");

const revealSection = () => {

    const trigger = window.innerHeight * 0.85;

    reveals.forEach(section=>{

        const top = section.getBoundingClientRect().top;

        if(top < trigger){

            section.classList.add("show");

        }

    });

}

window.addEventListener("scroll", revealSection);

revealSection();

const form = document.getElementById("recommendForm");

const loading = document.getElementById("loading");

const resultSection = document.getElementById("result");

const recommendationContainer =
document.getElementById("recommendation-container");

console.log("form :", form);
console.log("loading :", loading);
console.log("resultSection :", resultSection);
console.log("recommendationContainer :", recommendationContainer);

form.addEventListener("submit", async (e)=>{

    e.preventDefault();

    loading.style.display="block";

    resultSection.style.display="none";

    recommendationContainer.innerHTML="";

    const aktivitas =
    document.getElementById("activity").value;

    const undertone =
    document.getElementById("undertone").value;

    const hijab =
    document.getElementById("hijab").value;

    try{

        const response = await fetch("/recommend",{

            method:"POST",

            headers:{

                "Content-Type":"application/json"

            },

            body:JSON.stringify({

                aktivitas,

                undertone,

                hijab

            })

        });

        const results = await response.json();

        loading.style.display = "none";

        resultSection.style.display = "block";

        recommendationContainer.innerHTML = "";

        const badges = [
            "🥇 Best Match",
            "🥈 Runner Up",
            "🥉 Third Choice"
        ];

        results.slice(0, 3).forEach((item, index) => {

            recommendationContainer.innerHTML += `

                <div class="result-card">

                    <img
                        src="/images/${item.nama_gambar}"
                        alt="Recommended Outfit">

                    <div class="result-content">

                        <span class="result-badge">
                            ${badges[index]}
                        </span>

                        <h3>Outfit Recommendation</h3>

                        <p>Similarity Score</p>

                        <h2>${item.score.toFixed(2)}</h2>

                    </div>

                </div>

            `;

        });
    }

    catch(error){

        console.error(error);

        loading.style.display="none";

        alert("Recommendation failed.");

    }

});


