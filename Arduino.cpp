int buzzer = 8;

void setup()
{
    pinMode(buzzer, OUTPUT);
    Serial.begin(9600);
}

void loop()
{
    if (Serial.available() > 0)
    {
        char data = Serial.read();
        if (data == '1')
        {
            digitalWrite(buzzer, HIGH);
        }
        else if (data == '0')
        {
            digitalWrite(buzzer, LOW);
        }
    }
}
