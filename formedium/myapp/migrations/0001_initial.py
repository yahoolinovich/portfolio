# Generated by Django 4.1.4 on 2023-01-05 20:09

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.ImageField(upload_to='images/')),
                ('caption', models.CharField(blank=True, max_length=255)),
            ],
        ),
    ]
