using System.Collections;
using HKRL.Utils;
using HKRL.Game;
using InControl;
using Modding;

using UnityEngine;

namespace HKRL.Environments
{
	public class MultiEnv : WebsocketEnv
	{
		public (int, int) ObservationSize;
		public int ActionSize;
		public string Level;
		private int hitsTakenInStep = 0;
		private int damageDoneInStep = 0;
		private bool bossWouldDieInStep = false;
		public int frameSkipCount;
		public int TimeScale = 1;
		private HitboxReaderHook hitboxReaderHook = new();
		private Game.InputDeviceShim inputDeviceShim = new();
		private TimeScale timeManager;

		public MultiEnv(string url, params string[] protocols) : base(url, protocols)
		{
			Connect();
			// yield return new Utils.Socket.WaitForMessage(socket);
			var message = socket.UnreadMessages.Dequeue();
			if (message.type == "load")
			{

			}

		}

		protected override IEnumerator OnMessage(Message message)
		{
			switch (message.type)
			{
				case "close":
					_terminate = true;
					break;
				case "action":
					yield return Step(message.data);
					break;
				case "init":
					yield return Init(message.data);
					break;
				case "pause":
					yield return Pause(message.data);
					break;
				case "resume":
					yield return Resume(message.data);
					break;
				default:
					break;
			}

		}

		private IEnumerator Pause(MessageData data)
		{
			Time.timeScale = 0;
			SendMessage(new Message
			{
				type = "pause",
				data = data
			});
			yield break;
		}

		private IEnumerator Resume(MessageData data)
		{
			Time.timeScale = TimeScale;
			SendMessage(new Message
			{
				type = "resume",
				data = data
			});
			yield break;
		}

		private IEnumerator Init(MessageData data)
		{
			ObservationSize = (data.state_size[0], data.state_size[1]);
			ActionSize = data.action_size.Value;
			Level = data.level;
			frameSkipCount = data.frames_per_wait.Value;
			TimeScale = data.time_scale.Value;
			

			// yield return LoadLevel(Level);
			yield return SceneHooks.LoadBossScene(Level);
			ModHooks.AfterTakeDamageHook += SetZeroDamage;
			On.HealthManager.TakeDamage += SetHitsDealtInStep;
			timeManager = new(TimeScale);

			Image observation = GetObservation();
			data.state = observation.ToIntArray();

			SendMessage(new Message
			{
				type = "init",
				data = data
			});

			yield break;

		}

		private int SetZeroDamage(int damageType, int _){
			hitsTakenInStep++;
			return 0;
		}

		private void SetHitsDealtInStep(On.HealthManager.orig_TakeDamage orig, HealthManager self, HitInstance hitInstance)
		{
			damageDoneInStep += hitInstance.DamageDealt;
			bossWouldDieInStep = SceneHooks.ResetBossHealthAfterThreshold(orig, self, hitInstance, 50, 800);
		}

		private IEnumerator LoadLevel(string level)
		{
			yield break;
		}

		private Image GetObservation()
		{
			Image image = new(Screen.width, Screen.height);
			var hitboxes = hitboxReaderHook.GetHitboxes();
			Camera main = Camera.main;
			foreach (var hitboxClass in hitboxes)
			{
				byte label = (byte)(hitboxClass.Key + 1);
				foreach (var hitbox in hitboxClass.Value)
				{
					HitboxReader.RenderHitbox(hitbox, main, label, image);
				}
			}
			image = image.ResizeNearestNeighbor(ObservationSize.Item1, ObservationSize.Item2);
			return image;
		}



		private IEnumerator Step(MessageData data)
		{
			ActionCodeToInput(data.action.Value);
			for (int i = 0; i < frameSkipCount; i++)
			{
				yield return null;
			}
			data.reward = (damageDoneInStep / 4) - (50 * hitsTakenInStep);
			data.done = bossWouldDieInStep;
			// inputDeviceShim.Reset();
			Image observation = GetObservation();
			data.state = observation.ToIntArray();
			
			SendMessage(new Message
			{
				type = "step",
				data = data
			});

			hitsTakenInStep = 0;
			damageDoneInStep = 0;
			bossWouldDieInStep = false;

			yield break;
		}

		protected override IEnumerator Setup()
		{
			hitboxReaderHook.Load();
			InputManager.AttachDevice(inputDeviceShim);
			yield break;
		}

		protected override IEnumerator Dispose()
		{
			ModHooks.AfterTakeDamageHook -= SetZeroDamage;
			On.HealthManager.TakeDamage -= SetHitsDealtInStep;
			timeManager.Dispose();
			InputManager.DetachDevice(inputDeviceShim);
			hitboxReaderHook.Unload();
			CloseSocket();
			yield break;
		}

		private void ActionCodeToInput(int actionCode)
		{
			string action = actionCode.IntToBaseThreeString(4);
			switch (action[0])
			{
				case '0':
					inputDeviceShim.Left();
					break;
				case '1':
					inputDeviceShim.Right();
					break;
				default:
					inputDeviceShim.StopLR();
					break;
			}

			switch (action[1])
			{
				case '0':
					inputDeviceShim.Up();
					break;
				case '1':
					inputDeviceShim.Down();
					break;
				default:
					inputDeviceShim.StopUD();
					break;
			}

			switch (action[2])
			{
				case '0':
					inputDeviceShim.Attack();
					break;
				case '1':
					inputDeviceShim.Cast();
					break;
				default:
					inputDeviceShim.StopAC();
					break;
			}

			switch (action[3])
			{
				case '0':
					inputDeviceShim.Jump();
					break;
				case '1':
					inputDeviceShim.Dash();
					break;
				default:
					inputDeviceShim.StopJD();
					break;
			}
		}


	}
}