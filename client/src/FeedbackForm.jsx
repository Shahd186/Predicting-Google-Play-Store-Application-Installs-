import { useState } from "react";

export default function FeedbackForm({ apiUrl, predictionId }) {
    const [value, setValue] = useState("");
    const [status, setStatus] = useState("idle");
    const [message, setMessage] = useState("");

    const sending = status === "sending";
    const finished = status === "saved" || status === "duplicate";

    async function submitFeedback(event) {
        event.preventDefault();
        if (sending || finished) return;

        const count = Number(value);
        if (value.trim() === "" || !Number.isSafeInteger(count) || count < 0) {
            setStatus("invalid");
            setMessage("Enter a whole number of installs, zero or more.");
            return;
        }

        setStatus("sending");
        setMessage("");

        try {
            const response = await fetch(`${apiUrl}/feedback`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    prediction_id: predictionId,
                    actual_installs: count,
                }),
            });

            if (response.ok) {
                setStatus("saved");
                setMessage(`Feedback saved: ${count.toLocaleString("en-US")} actual installs.`);
                return;
            }

            const messages = {
                404: "This prediction could not be found. Make a new prediction and try again.",
                409: "Actual installs have already been reported for this prediction.",
                422: "The server could not accept this report. Check the install count.",
                429: "Too many reports were sent. Wait a moment and try again.",
                503: "The service is temporarily unavailable. Try again shortly.",
            };

            setStatus(response.status === 409 ? "duplicate" : "error");
            setMessage(messages[response.status] ?? "Could not save feedback. Please try again.");
        } catch {
            setStatus("error");
            setMessage("Could not reach the server. Check the connection and try again.");
        }
    }

    return (
        <section aria-labelledby="feedback-title" className="card feedback">
            <h3 id="feedback-title">Report actual installs</h3>

            <form onSubmit={submitFeedback} noValidate>
                <label htmlFor="actual-installs">Actual number of installs </label>
                <input
                    id="actual-installs"
                    type="number"
                    min="0"
                    step="1"
                    value={value}
                    disabled={sending || finished}
                    aria-invalid={status === "invalid"}
                    aria-describedby="feedback-message"
                    onChange={(event) => {
                        setValue(event.target.value);
                        setStatus("idle");
                        setMessage("");
                    }}
                />

                <button type="submit" disabled={sending || finished}>
                    {sending ? "Sending..." : finished ? "Feedback recorded" : "Send feedback"}
                </button>

                <p id="feedback-message" role="status">{message}</p>
            </form>
        </section>
    );
}