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
		private int totalDamageTaken = 0;
		private int bossHP = 0;
		private bool bossWouldDieInStep = false;
		private const int PLAYERHEALTH = 8;
		public int frameSkipCount;
		public bool FrameStack;
		public int TimeScale = 1;
		public int[] FrameStackBuffer;
		private HitboxReaderHook hitboxReaderHook = new();
		private Game.InputDeviceShim inputDeviceShim = new();
		private TimeScale timeManager;

		public MultiEnv(string url, params string[] protocols) : base(url, protocols)
		{

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
				case "reset":
					yield return Reset(message.data);
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

		private IEnumerator Reset(MessageData data)
		{
			ObservationSize = (data.state_size[0], data.state_size[1]);
			ActionSize = data.action_size.Value;
			Level = data.level;
			frameSkipCount = data.frames_per_wait.Value;
			TimeScale = data.time_scale.Value;
			totalDamageTaken = 0;
			// FrameStack = data.frame_stack.Value;
			

			// yield return LoadLevel(Level);
			yield return SceneHooks.LoadBossScene(Level);
			bossHP = BossSceneController.Instance.bosses[0].hp;
			ModHooks.AfterTakeDamageHook -= SetZeroDamage;
			On.HealthManager.TakeDamage -= SetHitsDealtInStep;

			ModHooks.AfterTakeDamageHook += SetZeroDamage;
			On.HealthManager.TakeDamage += SetHitsDealtInStep;
			timeManager = new(TimeScale);

			Image observation = GetObservation();
			data.state = observation.ToIntArray();
			data.boss_damage_taken = damageDoneInStep;
			data.player_hits_taken = hitsTakenInStep;
			data.boss_total_health = bossHP;
			data.player_total_health = PLAYERHEALTH;

			SendMessage(new Message
			{
				type = "reset",
				data = data
			});

			yield break;

		}

		private int SetZeroDamage(int damageType, int _){
			hitsTakenInStep++;
			totalDamageTaken ++;
			if (totalDamageTaken > PLAYERHEALTH) {
				return 0;
			}
			return 1; // for videos
			// return 0;
		}

		private void SetHitsDealtInStep(On.HealthManager.orig_TakeDamage orig, HealthManager self, HitInstance hitInstance)
		{
			damageDoneInStep += hitInstance.DamageDealt;
			bossWouldDieInStep = SceneHooks.ResetBossHealthAfterThreshold(orig, self, hitInstance, -100, bossHP);
			orig(self, hitInstance);
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
			// hitsTakenInStep = 0;
			// damageDoneInStep = 0;
			// bossWouldDieInStep = false;
			ActionCodeToInput(data.action.Value);
			for (int i = 0; i < frameSkipCount; i++)
			{
				yield return null;
			}
			data.reward = (((float)damageDoneInStep * PLAYERHEALTH) / ((float)bossHP + 1e-8f)) - (1 * hitsTakenInStep); // regularize between -1 and 1 (mostly)
			data.done = bossWouldDieInStep || totalDamageTaken > PLAYERHEALTH;
			// inputDeviceShim.Reset();
			Image observation = GetObservation();
			data.state = observation.ToIntArray();
			data.boss_damage_taken = damageDoneInStep;
			data.player_hits_taken = hitsTakenInStep;
			data.boss_total_health = bossHP;
			data.player_total_health = PLAYERHEALTH;
			
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
			Connect();
			yield return new Utils.Socket.WaitForMessage(socket);
			Message message = socket.UnreadMessages.Dequeue();
			// yield return OnMessage(message);
			if (message.type != "init")
			{
				yield return Setup();
				yield break;

			}
			On.GameManager.SaveGame += Game.SaveFileProxy.DisableSaveGame;
			Game.SaveFileProxy.LoadCompletedSave();
			// yield return new WaitForSeconds(5f);
			GameManager.instance.ContinueGame();
			yield return new Game.SceneHooks.WaitForSceneLoad("GG_Workshop"); 
			// yield return new Game.SceneHooks.WaitForSceneLoad("GG_Workshop"); 
			yield return new WaitForFinishedEnteringScene();
			yield return new WaitForSeconds(2f); // Workaround till I find a better solution 
			// HKRL.Instance.Log("Level loaded");

			hitboxReaderHook.Load();
			InputManager.AttachDevice(inputDeviceShim);
			SendMessage(message);
			// yield break;
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