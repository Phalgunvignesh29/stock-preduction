"use client";

import { useState, useRef, ChangeEvent } from "react";
import { Button } from "@workspace/ui/components/button";
import { Input } from "@workspace/ui/components/input";
import { Label } from "@workspace/ui/components/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@workspace/ui/components/select";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@workspace/ui/components/card";

type Signal = {
  type: "BUY" | "SELL";
  x_percent: number;
  y_percent: number;
  reason: string;
};

type KeyLevel = {
  type: "SUPPORT" | "RESISTANCE";
  description: string;
};

type Message = {
  role: "user" | "assistant";
  content: string;
};

type ApiResponse = {
  status: string;
  strategy: string;
  filename: string;
  trend: string;
  confidence: number;
  risk_level: string;
  summary: string;
  analysis: string;
  signals: Signal[];
  key_levels: KeyLevel[];
  strategy_notes: string;
} | null;

export default function Page() {
  const [strategy, setStrategy] = useState("Smart Money Concepts");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [data, setData] = useState<ApiResponse>(null);
  const [error, setError] = useState("");

  // Chat States
  const [messages, setMessages] = useState<Message[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [isChatLoading, setIsChatLoading] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setData(null);
      setError("");
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setIsLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("strategy", strategy);

      const res = await fetch("http://127.0.0.1:8006/api/analyze-graph", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail || `Server error ${res.status}`);
      }
      const json = await res.json();
      setData(json);
    } catch (err: any) {
      setError(err.message || "An error occurred.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleChat = async () => {
    if (!chatInput.trim() || !data) return;

    const userMessage: Message = { role: "user", content: chatInput };
    setMessages((prev) => [...prev, userMessage]);
    setChatInput("");
    setIsChatLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:8006/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage.content,
          history: messages,
          context: data.analysis, // Provide the chart analysis as context
        }),
      });

      if (!res.ok) throw new Error("Chat failed");
      const json = await res.json();
      const assistantMessage: Message = {
        role: "assistant",
        content: json.response,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setError("Chat error. Is Ollama running?");
    } finally {
      setIsChatLoading(false);
    }
  };

  const trendColor =
    data?.trend === "BULLISH"
      ? "text-green-700 bg-green-50 border-green-300"
      : data?.trend === "BEARISH"
        ? "text-red-700 bg-red-50 border-red-300"
        : "text-yellow-700 bg-yellow-50 border-yellow-300";

  return (
    <div className="flex h-screen bg-[#f5f5f5] text-[#111] font-[Inter,system-ui,sans-serif] antialiased overflow-hidden">
      {/* ─── Sidebar ─── */}
      <aside className="w-[320px] bg-white border-r border-gray-200 p-6 flex flex-col gap-6 shrink-0 overflow-y-auto">
        <div>
          <h1 className="text-xl font-bold tracking-tight">Chart Analyst</h1>
          <p className="text-xs text-gray-500 mt-1">
            Upload any stock/crypto chart screenshot. A real AI vision model
            reads the image and tells you where to buy and sell.
          </p>
        </div>

        {/* Strategy */}
        <div className="space-y-1.5">
          <Label className="text-xs font-semibold text-gray-600 uppercase tracking-wider">
            Analysis Strategy
          </Label>
          <Select value={strategy} onValueChange={setStrategy}>
            <SelectTrigger className="rounded-md border border-gray-300 h-9 text-sm focus-visible:ring-1 focus-visible:ring-blue-500">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="rounded-md border border-gray-200">
              <SelectItem value="Smart Money Concepts">
                Smart Money Concepts
              </SelectItem>
              <SelectItem value="Elliot Wave Theory">
                Elliot Wave Theory
              </SelectItem>
              <SelectItem value="Wyckoff Method">Wyckoff Method</SelectItem>
              <SelectItem value="ICT / Order Blocks">
                ICT / Order Blocks
              </SelectItem>
              <SelectItem value="Price Action + S/R">
                Price Action + Support/Resistance
              </SelectItem>
              <SelectItem value="Volume Profile Analysis">
                Volume Profile Analysis
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* File Upload */}
        <div className="space-y-1.5">
          <Label className="text-xs font-semibold text-gray-600 uppercase tracking-wider">
            Chart Image
          </Label>
          <div
            className="border-2 border-dashed border-gray-300 rounded-md p-4 flex flex-col items-center justify-center cursor-pointer hover:border-blue-400 hover:bg-blue-50/50 transition-all h-28"
            onClick={() => fileInputRef.current?.click()}
          >
            {file ? (
              <p className="text-xs font-medium text-gray-700 text-center truncate max-w-full">
                {file.name}
              </p>
            ) : (
              <>
                <svg
                  className="w-6 h-6 text-gray-400 mb-1"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="1.5"
                    d="M12 16v-8m0 0l-3 3m3-3l3 3M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14"
                  />
                </svg>
                <p className="text-xs text-gray-500">Click to upload</p>
              </>
            )}
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              accept="image/*"
              onChange={handleFileChange}
            />
          </div>
        </div>

        <Button
          onClick={handleAnalyze}
          disabled={isLoading || !file}
          className="rounded-md bg-[#111] hover:bg-[#333] text-white font-semibold h-10 text-sm disabled:opacity-40"
        >
          {isLoading ? "Analyzing..." : "Analyze Chart"}
        </Button>

        {error && (
          <div className="text-red-600 text-xs font-medium bg-red-50 border border-red-200 rounded-md p-3">
            {error}
          </div>
        )}

        <div className="mt-auto pt-4 border-t border-gray-100">
          <p className="text-[10px] text-gray-400">
            Powered by local LLaVA Vision Model
          </p>
        </div>
      </aside>

      {/* ─── Main Content ─── */}
      <main className="flex-1 overflow-y-auto p-6 flex gap-6">
        {previewUrl ? (
          <>
            {/* Chart Canvas with Overlays */}
            <div className="flex-1 flex flex-col min-w-0">
              <div className="bg-white border border-gray-200 rounded-lg shadow-sm flex-1 flex flex-col overflow-hidden">
                <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between">
                  <h2 className="text-sm font-bold text-gray-800">
                    Chart View
                  </h2>
                  {data && (
                    <span
                      className={`text-xs font-bold px-2.5 py-1 rounded border ${trendColor}`}
                    >
                      {data.trend} · Confidence {data.confidence}/10 · Risk:{" "}
                      {data.risk_level}
                    </span>
                  )}
                </div>

                <div className="flex-1 relative flex items-center justify-center p-4 bg-gray-50">
                  <div className="relative inline-block">
                    <img
                      src={previewUrl}
                      alt="Uploaded chart"
                      className="max-w-full max-h-[65vh] object-contain rounded shadow-sm"
                    />

                    {/* Real AI signals overlaid */}
                    {data?.signals.map((signal, i) => (
                      <div
                        key={i}
                        className="absolute z-20 flex flex-col items-center"
                        style={{
                          left: `${signal.x_percent}%`,
                          top: `${signal.y_percent}%`,
                          transform: "translate(-50%, -50%)",
                        }}
                      >
                        <div
                          className={`px-2 py-0.5 rounded text-[10px] font-black uppercase shadow-md border-2 whitespace-nowrap ${
                            signal.type === "BUY"
                              ? "bg-green-500 text-white border-green-700"
                              : "bg-red-500 text-white border-red-700"
                          }`}
                        >
                          {signal.type}
                        </div>
                        <div
                          className={`w-0.5 h-3 ${signal.type === "BUY" ? "bg-green-500" : "bg-red-500"}`}
                        />
                        <div
                          className={`w-2.5 h-2.5 rounded-full border-2 border-white shadow-lg ${
                            signal.type === "BUY"
                              ? "bg-green-500"
                              : "bg-red-500"
                          }`}
                        />
                      </div>
                    ))}

                    {/* Loading scanner */}
                    {isLoading && (
                      <div className="absolute inset-0 bg-black/5 flex items-center justify-center rounded">
                        <div className="absolute w-full h-[2px] bg-blue-500 animate-pulse top-1/2 shadow-[0_0_15px_rgba(59,130,246,0.6)]" />
                        <span className="bg-white/90 text-xs font-semibold px-3 py-1.5 rounded shadow text-gray-700">
                          Gemini is reading the chart...
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Analysis Panel */}
            {data && (
              <div className="w-[380px] shrink-0 overflow-y-auto flex flex-col gap-4">
                {/* Summary */}
                <Card className="rounded-lg border border-gray-200 shadow-sm">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-bold text-gray-600 uppercase tracking-wider">
                      Verdict
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-base font-semibold text-gray-900 leading-relaxed">
                      {data.summary}
                    </p>
                  </CardContent>
                </Card>

                {/* Full Analysis */}
                <Card className="rounded-lg border border-gray-200 shadow-sm">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-bold text-gray-600 uppercase tracking-wider">
                      Detailed Analysis
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-gray-800 leading-[1.7] whitespace-pre-wrap">
                      {data.analysis}
                    </p>
                  </CardContent>
                </Card>

                {/* Key Levels */}
                {data.key_levels && data.key_levels.length > 0 && (
                  <Card className="rounded-lg border border-gray-200 shadow-sm">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-bold text-gray-600 uppercase tracking-wider">
                        Key Price Levels
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      {data.key_levels.map((level, i) => (
                        <div
                          key={i}
                          className="flex items-start gap-2 text-sm"
                        >
                          <span
                            className={`text-[10px] font-bold px-1.5 py-0.5 rounded mt-0.5 shrink-0 ${
                              level.type === "SUPPORT"
                                ? "bg-green-100 text-green-700"
                                : "bg-red-100 text-red-700"
                            }`}
                          >
                            {level.type}
                          </span>
                          <span className="text-gray-700">
                            {level.description}
                          </span>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                )}

                {/* Signal Log */}
                <Card className="rounded-lg border border-gray-200 shadow-sm">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-bold text-gray-600 uppercase tracking-wider">
                      Signal Log
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {data.signals.map((s, i) => (
                      <div
                        key={i}
                        className="bg-gray-50 border border-gray-200 rounded-md p-3"
                      >
                        <div className="flex justify-between items-center mb-1">
                          <span
                            className={`text-xs font-bold ${s.type === "BUY" ? "text-green-600" : "text-red-600"}`}
                          >
                            {s.type}
                          </span>
                        </div>
                        <p className="text-xs text-gray-600 leading-relaxed">
                          {s.reason}
                        </p>
                      </div>
                    ))}
                  </CardContent>
                </Card>

                {/* Strategy Notes */}
                {data.strategy_notes && (
                  <Card className="rounded-lg border border-gray-200 shadow-sm">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-bold text-gray-600 uppercase tracking-wider">
                        {data.strategy} Notes
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-gray-700 leading-[1.7]">
                        {data.strategy_notes}
                      </p>
                    </CardContent>
                  </Card>
                )}

                {/* ─── NEW: Chat with AI ─── */}
                <Card className="rounded-lg border-2 border-blue-100 shadow-md bg-blue-50/30 flex flex-col min-h-[400px]">
                  <CardHeader className="pb-2 border-b border-blue-100">
                    <CardTitle className="text-xs font-bold text-blue-600 uppercase tracking-wider flex items-center gap-2">
                       <span className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                       Chat with Catalyst AI
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="flex-1 flex flex-col p-0 overflow-hidden">
                    {/* Message List */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-4 max-h-[300px]">
                      {messages.length === 0 && (
                        <p className="text-xs text-blue-400 italic text-center mt-4">
                          Ask me anything about this chart...
                        </p>
                      )}
                      {messages.map((m, i) => (
                        <div
                          key={i}
                          className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
                        >
                          <div
                            className={`max-w-[85%] rounded-lg px-3 py-2 text-xs ${
                              m.role === "user"
                                ? "bg-blue-600 text-white rounded-tr-none"
                                : "bg-white border border-blue-100 text-gray-800 rounded-tl-none shadow-sm"
                            }`}
                          >
                            {m.content}
                          </div>
                        </div>
                      ))}
                      {isChatLoading && (
                        <div className="flex justify-start">
                          <div className="bg-white border border-blue-100 rounded-lg px-3 py-2 shadow-sm">
                            <div className="flex gap-1">
                               <div className="w-1 h-1 bg-blue-400 rounded-full animate-bounce" />
                               <div className="w-1 h-1 bg-blue-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                               <div className="w-1 h-1 bg-blue-400 rounded-full animate-bounce [animation-delay:0.4s]" />
                            </div>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Chat Input */}
                    <div className="p-3 bg-white border-t border-blue-100 flex gap-2">
                      <Input
                        placeholder="Ask a question..."
                        value={chatInput}
                        onChange={(e) => setChatInput(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleChat()}
                        className="text-xs h-8 border-blue-100 focus-visible:ring-blue-400"
                        disabled={isChatLoading}
                      />
                      <Button
                        size="sm"
                        onClick={handleChat}
                        disabled={isChatLoading || !chatInput.trim()}
                        className="h-8 bg-blue-600 hover:bg-blue-700 text-white"
                      >
                         Send
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </>
        ) : (
          <div className="m-auto text-center max-w-md">
            <svg
              className="w-16 h-16 mx-auto mb-4 text-gray-300"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="1"
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
            <h2 className="text-lg font-bold text-gray-600 mb-1">
              No chart uploaded yet
            </h2>
            <p className="text-sm text-gray-400">
              Upload a screenshot of any stock or crypto chart from TradingView,
              Binance, or any platform. The AI will read the actual image and
              tell you where to buy and sell.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
