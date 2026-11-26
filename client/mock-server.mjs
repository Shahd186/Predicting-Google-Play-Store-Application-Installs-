import { createServer } from "node:http";

const schema = {
    categories: ["Education"],
    content_ratings: ["Everyone"],
    android_versions: ["5.0"],
};

const examplePrediction = {
    tier: "Moderate",
    installs: [10001, 100000],
    confidence: {
        Low: 0.24,
        Moderate: 0.46,
        High: 0.22,
        "Very High": 0.08,
    },
    what_if: [
        { change: "ad_supported=false", high_or_above: "+0.06" },
    ],
    prediction_id: "p-1842",
    model_version: "rf-v1",
    observation_date: "2026-11-14",
    warnings: [],
};

function sendJson(response, status, body) {
    response.writeHead(status, {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    });
    response.end(JSON.stringify(body));
}

const server = createServer(async (request, response) => {
    if (request.method === "OPTIONS") {
        return sendJson(response, 204, null);
    }

    if (request.method === "GET" && request.url === "/schema") {
        return sendJson(response, 200, schema);
    }

    if (request.method === "POST" && request.url === "/predict") {
        let body = "";
        for await (const chunk of request) body += chunk;

        try {
            const app = JSON.parse(body);
            console.log("App received from React:", app);
            return sendJson(response, 200, {
                ...examplePrediction,
                what_if: [
                    {
                        change: `ad_supported=${!app.ad_supported}`,
                        high_or_above: app.ad_supported ? "+0.06" : "-0.06",
                    },
                    {
                        change: `free=${!app.free}`,
                        high_or_above: app.free ? "-0.005" : "+0.005",
                    },
                ],
                warnings: app.developer === "Example Dev"
                    ? []
                    : ["This developer was not seen in the training data."],
            });
        } catch {
            return sendJson(response, 422, { detail: "Invalid JSON." });
        }
    }

    return sendJson(response, 200, {
        ...examplePrediction,
        what_if: [
            {
                change: `ad_supported=${!app.ad_supported}`,
                high_or_above: app.ad_supported ? "+0.06" : "-0.06",
            },
            {
                change: `free=${!app.free}`,
                high_or_above: app.free ? "-0.005" : "+0.005",
            },
        ],
        warnings:
            app.developer === "Example Dev"
                ? []
                : ["This developer was not seen in the training data."],
    });
});

server.listen(8000, "127.0.0.1", () => {
    console.log("Mock API running at http://127.0.0.1:8000");
});