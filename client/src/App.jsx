import { useEffect, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

import FeedbackForm from "./FeedbackForm.jsx";

import "./App.css";

const initialForm = {
    category: "Education",
    developer: "Example Dev",
    content_rating: "Everyone",
    size_mb: "25",
    minimum_android: "5.0",
    free: true,
    ad_supported: true,
    privacy_policy: true,
    allow_feedback: false,
    released: "2024-01-15",
    last_updated: "2026-09-01",
};

export default function App() {
    const [schema, setSchema] = useState(null);
    const [form, setForm] = useState(initialForm);
    const [sizeVaries, setSizeVaries] = useState(false);
    const [errors, setErrors] = useState({});
    const [message, setMessage] = useState("");
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        async function loadSchema() {
            try {
                const response = await fetch(`${API_URL}/schema`);
                if (!response.ok) throw new Error();
                const apiSchema = await response.json();
                const allowed = (name) => apiSchema.fields?.find((field) => field.name === name)?.allowed;
                setSchema({
                    categories: allowed("category") ?? apiSchema.categories,
                    content_ratings: allowed("content_rating") ?? apiSchema.content_ratings,
                    android_versions: allowed("minimum_android") ?? apiSchema.android_versions,
                });
            } catch {
                setMessage("Could not load the form options. Check that the API is running, then refresh.");
            }
        }
        loadSchema();
    }, []);

    function updateField(event) {
        const { name, type, value, checked } = event.target;
        setForm((previous) => ({
            ...previous,
            [name]: type === "checkbox" ? checked : value,
        }));
        setErrors({});
        setResult(null);
        setMessage("");
    }

    async function predict(event) {
        event.preventDefault();

        const nextErrors = {};
        if (!sizeVaries && (form.size_mb === "" || Number(form.size_mb) < 0)) {
            nextErrors.size_mb = "Enter a size of 0 MB or more.";
        }
        if (form.last_updated < form.released) {
            nextErrors.last_updated = "Last update must be on or after release.";
        }
        setErrors(nextErrors);
        if (Object.keys(nextErrors).length > 0) return;

        setLoading(true);
        setMessage("");
        setResult(null);

        try {
            const response = await fetch(`${API_URL}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    ...form,
                    size_mb: sizeVaries ? null : Number(form.size_mb),
                }),
            });

            if (response.status === 503) {
                throw new Error("The model is loading. Try again shortly.");
            }
            if (!response.ok) {
                throw new Error(`The API could not process this request (${response.status}).`);
            }
            setResult(await response.json());
        } catch (error) {
            setMessage(
                error instanceof TypeError
                    ? "Could not reach the API. Check that the server is running."
                    : error.message
            );
        } finally {
            setLoading(false);
        }
    }

    return (
        <main>
            <h1>Google Play install predictor</h1>

            {message && <p role="alert">{message}</p>}

            {!schema && !message && <p>Loading form options...</p>}

            {schema && (
                <form onSubmit={predict}>
                    <fieldset disabled={loading}>
                        <legend>App</legend>

                        <p>
                            <label>
                                Category{" "}
                                <select name="category" value={form.category} onChange={updateField}>
                                    {schema.categories.map((option) => (
                                        <option key={option} value={option}>{option}</option>
                                    ))}
                                </select>
                            </label>
                        </p>

                        <p>
                            <label>
                                Developer name{" "}
                                <input name="developer" value={form.developer}
                                    onChange={updateField} required />
                            </label>
                        </p>

                        <p>
                            <label>
                                Content rating{" "}
                                <select name="content_rating" value={form.content_rating}
                                    onChange={updateField}>
                                    {schema.content_ratings.map((option) => (
                                        <option key={option} value={option}>{option}</option>
                                    ))}
                                </select>
                            </label>
                        </p>
                    </fieldset>

                    <fieldset disabled={loading}>
                        <legend>Monetisation</legend>

                        <p>
                            <label>
                                <input type="checkbox" name="free"
                                    checked={form.free} onChange={updateField} /> Free app
                            </label>
                        </p>

                        <p>
                            <label>
                                <input type="checkbox" name="ad_supported"
                                    checked={form.ad_supported} onChange={updateField} /> Contains ads
                            </label>
                        </p>
                    </fieldset>

                    <fieldset disabled={loading}>
                        <legend>Technical</legend>

                        <p>
                            <label>
                                Minimum Android version{" "}
                                <select name="minimum_android" value={form.minimum_android}
                                    onChange={updateField}>
                                    {schema.android_versions.map((option) => (
                                        <option key={option} value={option}>{option}</option>
                                    ))}
                                </select>
                            </label>
                        </p>

                        <p>
                            <label>
                                App size (MB){" "}
                                <input type="number" step="any" name="size_mb"
                                    value={sizeVaries ? "" : form.size_mb}
                                    disabled={sizeVaries}
                                    onChange={updateField}
                                    aria-invalid={Boolean(errors.size_mb)}
                                    aria-describedby={errors.size_mb ? "size-error" : undefined} />
                            </label>
                            {errors.size_mb && <span id="size-error" role="alert"> {errors.size_mb}</span>}
                        </p>

                        <p>
                            <label>
                                <input
                                    type="checkbox"
                                    checked={sizeVaries}
                                    onChange={(event) => {
                                        setSizeVaries(event.target.checked);
                                        setErrors((previous) => ({ ...previous, size_mb: undefined }));
                                        setResult(null);
                                        setMessage("");
                                    }}
                                /> Varies with device
                            </label>
                        </p>
                    </fieldset>

                    <fieldset disabled={loading}>
                        <legend>Timeline</legend>

                        <p>
                            <label>
                                Release date{" "}
                                <input type="date" name="released" value={form.released}
                                    onChange={updateField} required />
                            </label>
                        </p>

                        <p>
                            <label>
                                Last update{" "}
                                <input type="date" name="last_updated" value={form.last_updated}
                                    onChange={updateField} required
                                    aria-invalid={Boolean(errors.last_updated)}
                                    aria-describedby={errors.last_updated ? "date-error" : undefined} />
                            </label>
                            {errors.last_updated && (
                                <span id="date-error" role="alert"> {errors.last_updated}</span>
                            )}
                        </p>
                    </fieldset>

                    <fieldset disabled={loading}>
                        <legend>Policy</legend>

                        <p>
                            <label>
                                <input type="checkbox" name="allow_feedback"
                                    checked={form.allow_feedback} onChange={updateField} /> Let me report actual installs later
                            </label>
                        </p>

                        <p>
                            <label>
                                <input type="checkbox" name="privacy_policy"
                                    checked={form.privacy_policy} onChange={updateField} /> Has a privacy policy
                            </label>
                        </p>
                    </fieldset>

                    <button type="submit" disabled={loading}>
                        {loading ? "Predicting..." : "Predict installs"}
                    </button>
                </form>
            )}

            {result && (
                <section aria-live="polite">
                    <h2>Predicted tier: {result.tier}</h2>
                    <p>
                        {result.installs[0].toLocaleString("en-US")}
                        {" to "}
                        {result.installs[1] == null
                            ? "unlimited"
                            : result.installs[1].toLocaleString("en-US")} installs
                    </p>

                    <h3>Model confidence</h3>
                    {Object.entries(result.confidence).map(([tier, value]) => (
                        <p key={tier}>
                            <label>
                                {tier} <progress value={value} max="1" /> {Math.round(value * 100)}%
                            </label>
                        </p>
                    ))}
                    {result.what_if?.length > 0 && (
                        <section>
                            <h3>What if I change my app?</h3>
                            <p>Change in combined model confidence for High and Very High installs:</p>

                            <ul>
                                {result.what_if.map(({ change, high_or_above_delta, high_or_above }) => {
                                    const labels = {
                                        "ad_supported=false": "Turn ads off",
                                        "ad_supported=true": "Turn ads on",
                                        "free=false": "Make the app paid",
                                        "free=true": "Make the app free",
                                    };

                                    const points = Number(high_or_above_delta ?? high_or_above) * 100;
                                    const amount = Math.abs(points);
                                    const label = labels[change] ?? change;
                                    let description = "No notable effect";

                                    if (amount >= 1) {
                                        const direction = points > 0 ? "Increase" : "Decrease";
                                        const formatted = amount.toLocaleString("en-US", {
                                            maximumFractionDigits: 1,
                                        });
                                        description = `${direction} of ${formatted} percentage ${amount === 1 ? "point" : "points"
                                            }`;
                                    }

                                    return (
                                        <li key={change}>
                                            <strong>{label}:</strong> {description}
                                        </li>
                                    );
                                })}
                            </ul>

                            <small>
                                These comparisons show patterns learned from past apps.
                                Changing a setting does not guarantee a change in actual installs.
                            </small>
                        </section>
                    )}

                    {result.warnings?.length > 0 && (
                        <section>
                            <h3>Prediction warnings</h3>
                            <ul>
                                {result.warnings.map((warning) => (
                                    <li key={warning}>{warning}</li>
                                ))}
                            </ul>
                        </section>
                    )}
                    <p>Observation date: {result.observation_date}</p>
                    {(result.feedback_enabled ?? true) && <FeedbackForm
                        key={result.prediction_id}
                        apiUrl={API_URL}
                        predictionId={result.prediction_id}
                    />}
                </section>
            )}
        </main>
    );
}