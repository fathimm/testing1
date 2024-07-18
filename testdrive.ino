void loop() {
  // Read sensor 1
  digitalWrite(TRIGPIN1, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIGPIN1, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIGPIN1, LOW);
  timer1 = pulseIn(ECHOPIN1, HIGH);
  jarak1 = timer1 / 58;

  // Read sensor 2
  digitalWrite(TRIGPIN2, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIGPIN2, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIGPIN2, LOW);
  timer2 = pulseIn(ECHOPIN2, HIGH);
  jarak2 = timer2 / 58;

  // Read sensor 3
  digitalWrite(TRIGPIN3, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIGPIN3, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIGPIN3, LOW);
  timer3 = pulseIn(ECHOPIN3, HIGH);
  jarak3 = timer3 / 58;

  // Send data to Raspberry Pi
  Serial.print(jarak1);
  Serial.print(",");
  Serial.print(jarak2);
  Serial.print(",");
  Serial.println(jarak3);

  delay(1000);
}
