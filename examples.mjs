/* Resolve guided cases from the loaded text. No conversation text is bundled here. */
function sourceTurn(data, history, pattern, role) {
  for (const session of data.histories[history].sessions) {
    const turn = data.sessions[session].turns.findIndex(item => item.role === role && pattern.test(data.points[item.p].text));
    if (turn !== -1) return { kind: "turn", session, turn, p: data.sessions[session].turns[turn].p };
  }
  return null;
}

function storedMemory(data, history, system, pattern, exclude = /$^/) {
  const p = (data.histories[history].memories[system] || []).find(p => pattern.test(data.points[p].text) && !exclude.test(data.points[p].text));
  return p === undefined ? null : { kind: "memory", p };
}

function focusedExcerpt(data, anchor, pattern) {
  return data.points[anchor.p].text.match(pattern)?.[0].trim() || data.points[anchor.p].text;
}

export function guidedExamples(data) {
  const examples = [];
  const mileage = data.histories.findIndex(history => /miles per gallon.*compared to now/i.test(history.question));
  if (mileage !== -1) {
    const earlier = sourceTurn(data, mileage, /30 miles per gallon/i, "user");
    const later = sourceTurn(data, mileage, /28 miles per gallon/i, "user");
    const summary = storedMemory(data, mileage, "m2", /plans to get a tune-up/i, /\b30\b/);
    if (earlier) earlier.excerpt = focusedExcerpt(data, earlier, /[^.!?]*30 miles per gallon[^.!?]*[.!?]?/i);
    if (later) later.excerpt = focusedExcerpt(data, later, /[^.!?]*28 miles per gallon[^.!?]*[.!?]?/i);
    if (earlier && summary && data.systems.some(system => system.id === "m2" && system.label === "Ext-Dense")) examples.push({
      id: "lost-detail", label: "A detail disappears", history: mileage, system: "m2",
      title: "A summary drops a number",
      description: "The memory keeps the tune-up plan, but not the source's mileage value.",
      caveat: "One sampled memory; the full store may contain this value.",
      anchors: [{ ...earlier, label: "Source turn" }, { ...summary, label: "Saved memory" }],
    });
    if (earlier && later && earlier.session !== later.session) examples.push({
      id: "two-values", label: "Two sessions, two values", history: mileage, system: "",
      title: "Which value is current?",
      description: "The sessions report 30 and 28 mpg. Comparing them needs time context, not just similarity.",
      caveat: "Separate paths show separate sessions, not one continuous conversation.",
      anchors: [{ ...earlier, label: "Earlier report" }, { ...later, label: "Later report" }],
    });
  }
  const restaurant = data.histories.findIndex(history => /Italian restaurant in Rome.*recommended/i.test(history.question));
  if (restaurant !== -1) {
    const source = sourceTurn(data, restaurant, /Trattoria Da Enzo al 29[\s\S]*Roscioli/i, "assistant");
    const memory = storedMemory(data, restaurant, "m2", /Assistant recommended Trattoria Da Enzo al 29/i);
    if (source) source.excerpt = focusedExcerpt(data, source, /Trattoria Da Enzo al 29[^\n]*/i);
    if (source && memory && data.systems.some(system => system.id === "m2" && system.label === "Ext-Dense")) examples.push({
      id: "assistant-memory", label: "An assistant suggestion", history: restaurant, system: "m2",
      title: "Who said it?",
      description: "This is an assistant recommendation, not a user saying they visited or liked the restaurant.",
      caveat: "Text supports the correspondence. Extraction provenance and retrieval are not verified.",
      anchors: [{ ...source, label: "Assistant turn" }, { ...memory, label: "Saved memory" }],
    });
  }
  return examples;
}
