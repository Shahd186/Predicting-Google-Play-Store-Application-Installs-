import { useEffect, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const initialForm = {
    category: "Education",
    developer: "Example Dev",
    content_rating: "Everyone",
    size_mb: "25",
    minimum_android: "5.0",
    free: true,
    ad_supported: true,
    privacy_policy: true,
    released: "2024-01-15",
    last_updated: "2026-09-01",
};

export default function App() {
    const [schema, setSchema] = useState(null);
    const [form, setForm] = useState(initialForm);
    const [errors, setErrors] = useState({});
    const [message, setMessage] = useState("");
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        async function loadSchema() {
            try {
                const response = await fetch(`${API_URL}/schema`);
                if (!response.ok) throw new Error();
                setSchema(await response.json());
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
        if (form.size_mb === "" || Number(form.size_mb) < 0) {
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
                body: JSON.stringify({ ...form, size_mb: Number(form.size_mb) }),
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

            {!schema && !message && <p>Loading form options…</p>}

            {schema && (
                <form onSubmit={predict}>
                    <fieldset disabled={loading}>
                        <legend>App details</legend>

                        {[
                            ["category", "Category", schema.categories],
                            ["content_rating", "Content rating", schema.content_ratings],
                            ["minimum_android", "Minimum Android version", schema.android_versions],
                        ].map(([name, label, options]) => (
                            <p key={name}>
                                <label>
                                    {label}{" "}
                                    <select name={name} value={form[name]} onChange={updateField}>
                                        {options.map((option) => (
                                            <option key={option} value={option}>{option}</option>
                                        ))}
                                    </select>
                                </label>
                            </p>
                        ))}

                        <p>
                            <label>
                                Developer name{" "}
                                <input name="developer" value={form.developer}
                                    onChange={updateField} required />
                            </label>
                        </p>

                        <p>
                            <label>
                                App size (MB){" "}
                                <input type="number" step="any" name="size_mb"
                                    value={form.size_mb} onChange={updateField} />
                            </label>
                            {errors.size_mb && <span role="alert"> {errors.size_mb}</span>}
                        </p>

                        {[
                            ["free", "Free app"],
                            ["ad_supported", "Contains ads"],
                            ["privacy_policy", "Has a privacy policy"],
                        ].map(([name, label]) => (
                            <p key={name}>
                                <label>
                                    <input type="checkbox" name={name}
                                        checked={form[name]} onChange={updateField} /> {label}
                                </label>
                            </p>
                        ))}

                        {[
                            ["released", "Release date"],
                            ["last_updated", "Last update"],
                        ].map(([name, label]) => (
                            <p key={name}>
                                <label>
                                    {label}{" "}
                                    <input type="date" name={name} value={form[name]}
                                        onChange={updateField} required />
                                </label>
                                {errors[name] && <span role="alert"> {errors[name]}</span>}
                            </p>
                        ))}

                        <button type="submit">
                            {loading ? "Predicting…" : "Predict installs"}
                        </button>
                    </fieldset>
                </form>
            )}

            {result && (
                <section aria-live="polite">
                    <h2>Predicted tier: {result.tier}</h2>
                    <p>
                        {result.installs[0].toLocaleString("en-US")}–
                        {result.installs[1].toLocaleString("en-US")} installs
                    </p>

                    <h3>Model confidence</h3>
                    {Object.entries(result.confidence).map(([tier, value]) => (
                        <p key={tier}>
                            <label>
                                {tier} <progress value={value} max="1" /> {Math.round(value * 100)}%
                            </label>
                        </p>
                    ))}

                    <p>Observation date: {result.observation_date}</p>
                </section>
            )}
        </main>
    );
}