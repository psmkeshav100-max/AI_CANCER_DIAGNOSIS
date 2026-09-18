const form = document.getElementById("predictionForm");

const predictButton = document.getElementById("predictButton");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const error = document.getElementById("error");

const diagnosis = document.getElementById("diagnosis");
const malignantProbability = document.getElementById("malignantProbability");
const benignProbability = document.getElementById("benignProbability");

const API_URL = "https://ai-cancer-diagnosis-5.onrender.com/predict";

form.addEventListener("submit", async function (event) {

    event.preventDefault();

    result.classList.add("hidden");
    error.classList.add("hidden");

    predictButton.disabled = true;
    predictButton.textContent = "Predicting...";

    loading.classList.remove("hidden");

    try {

        const featureIds = [
            "mean_radius",
            "mean_texture",
            "mean_perimeter",
            "mean_area",
            "mean_smoothness",
            "mean_compactness",
            "mean_concavity",
            "mean_concave_points",
            "mean_symmetry",
            "mean_fractal_dimension",

            "radius_error",
            "texture_error",
            "perimeter_error",
            "area_error",
            "smoothness_error",
            "compactness_error",
            "concavity_error",
            "concave_points_error",
            "symmetry_error",
            "fractal_dimension_error",

            "worst_radius",
            "worst_texture",
            "worst_perimeter",
            "worst_area",
            "worst_smoothness",
            "worst_compactness",
            "worst_concavity",
            "worst_concave_points",
            "worst_symmetry",
            "worst_fractal_dimension"
        ];

        const features = featureIds.map(function (id) {

            const input = document.getElementById(id);

            return Number(input.value);

        });

        const response = await fetch(API_URL, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                features: features
            })

        });

        if (!response.ok) {

            const errorData = await response.json();

            throw new Error(
                errorData.detail || "Prediction request failed."
            );

        }

        const data = await response.json();

        diagnosis.textContent = data.diagnosis;

        malignantProbability.textContent =
            `${data.malignant_probability}%`;

        benignProbability.textContent =
            `${data.benign_probability}%`;

        result.classList.remove("hidden");

    } catch (err) {

        console.error("Prediction Error:", err);

        error.textContent =
            `Error: ${err.message}`;

        error.classList.remove("hidden");

    } finally {

        loading.classList.add("hidden");

        predictButton.disabled = false;

        predictButton.textContent = "Predict Diagnosis";

    }

});